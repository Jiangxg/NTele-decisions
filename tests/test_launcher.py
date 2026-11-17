import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from ntele_decision.core import Settings
from ntele_decision.launcher import ManagedSGLang, ManagedVLLM


class LauncherTests(unittest.TestCase):
    def test_backend_log_is_private_and_rejects_symlinks(self):
        from ntele_decision.launcher import open_backend_log
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "backend.log"
            with open_backend_log(path) as log:
                log.write("first\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            path.chmod(0o644)
            with open_backend_log(path) as log:
                log.write("second\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.read_text(), "first\nsecond\n")
            link = Path(directory) / "link.log"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                open_backend_log(link)

    def test_cli_vllm_launch_and_failure_cleanup(self):
        from ntele_decision.server import main
        argv = ['ntele-decide', 'serve', '--launch-vllm', '--model-path', '/weights',
                '--model', 'decision-lora', '--served-model-name', 'chat',
                '--vllm-arg=--enable-lora']
        with patch.object(sys, 'argv', argv), patch('ntele_decision.launcher.ManagedVLLM') as launcher, patch('ntele_decision.server.create_app') as app, patch('uvicorn.run') as run:
            main()
            cfg = launcher.call_args.args[1]
            self.assertEqual(cfg.backend, 'vllm')
            self.assertEqual(cfg.model, 'decision-lora')
            self.assertEqual(launcher.call_args.kwargs['served_model_name'], 'chat')
            launcher.return_value.start.assert_called_once()
            launcher.return_value.wait_ready.assert_called_once()
            launcher.return_value.stop.assert_called_once()
            app.assert_called_once_with(cfg, verify_backend=True)
            run.assert_called_once()
        with patch.object(sys, 'argv', argv), patch('ntele_decision.launcher.ManagedVLLM') as launcher, patch('uvicorn.run') as run:
            launcher.return_value.wait_ready.side_effect = RuntimeError('failed startup')
            with self.assertRaisesRegex(RuntimeError, 'failed startup'):
                main()
            launcher.return_value.stop.assert_called_once()
            run.assert_not_called()

    def test_cli_rejects_conflicting_launch_settings(self):
        from ntele_decision.server import main
        for flags in [
            ['--launch-vllm'],
            ['--launch-vllm', '--model-path', '/weights', '--backend-url', 'http://external'],
            ['--launch-vllm', '--model-path', '/weights', '--backend', 'sglang'],
            ['--launch-vllm', '--model-path', '/weights', '--temperature', '1'],
            ['--vllm-arg=--enable-lora'],
        ]:
            with self.subTest(flags=flags), patch.object(sys, 'argv', ['ntele-decide', 'serve', *flags]), patch('sys.stderr'):
                with self.assertRaises(SystemExit) as result:
                    main()
                self.assertEqual(result.exception.code, 2)

    def test_vllm_launches_base_with_decision_adapter(self):
        owner = ManagedVLLM('/weights', Settings(backend='vllm', model='decision-lora'),
            gpus='2,3', served_model_name='qwen-chat',
            extra=['--enable-lora', '--lora-modules', 'decision-lora=/adapter'])
        self.assertIn('vllm.entrypoints.cli.main', owner.command)
        self.assertEqual(owner.command[owner.command.index('--served-model-name') + 1], 'qwen-chat')
        self.assertEqual(owner.command[owner.command.index('--tensor-parallel-size') + 1], '2')
        self.assertEqual(owner.command[-3:], ['--enable-lora', '--lora-modules', 'decision-lora=/adapter'])
        with patch('ntele_decision.launcher.importlib.metadata.version', return_value='0.33.0rc3') as version, patch('ntele_decision.launcher.socket.socket'), patch('ntele_decision.launcher.open_backend_log'), patch('ntele_decision.launcher.subprocess.Popen') as popen, patch('ntele_decision.launcher.os.killpg') as kill:
            owner.start()
            version.assert_called_once_with('vllm')
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2,3')
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            owner.stop()
            kill.assert_called_once()

    def test_vllm_health_and_early_exit(self):
        owner = ManagedVLLM('/weights', Settings(backend='vllm'))
        owner.process = MagicMock()
        owner.process.poll.return_value = None
        with patch('ntele_decision.launcher.httpx.Client') as client:
            client.return_value.__enter__.return_value.get.return_value.is_success = True
            owner.wait_ready()
        owner.process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'vLLM exited'):
            owner.wait_ready()

    def test_vllm_rejects_owned_arguments_and_zero_tp(self):
        for extra in [['--port=9000'], ['--served-model-name', 'other'], ['-tp', '2']]:
            with self.assertRaises(ValueError):
                ManagedVLLM('/weights', Settings(backend='vllm'), extra=extra)
        with self.assertRaises(ValueError):
            ManagedVLLM('/weights', Settings(backend='vllm'), tp=0)

    def test_command_and_ownership(self):
        owner = ManagedSGLang('/weights', Settings(), gpus='2,3')
        self.assertEqual(owner.command[-2:], ['--tp', '2'])
        with patch('ntele_decision.launcher.importlib.metadata.version', return_value='0.5.21'), patch('ntele_decision.launcher.socket.socket'), patch('ntele_decision.launcher.open_backend_log'), patch('ntele_decision.launcher.subprocess.Popen') as popen, patch('ntele_decision.launcher.os.killpg') as kill:
            owner.start()
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2,3')
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            owner.stop()
            kill.assert_called_once()

    def test_version_and_reserved_flags(self):
        with patch('ntele_decision.launcher.importlib.metadata.version', return_value='0.5.19'):
            with self.assertRaises(RuntimeError):
                ManagedSGLang('/weights', Settings()).start()
        with self.assertRaises(ValueError):
            ManagedSGLang('/weights', Settings(), extra=['--port=9000'])

    def test_early_exit(self):
        owner = ManagedSGLang('/weights', Settings())
        owner.process = MagicMock()
        owner.process.poll.return_value = 1
        with self.assertRaises(RuntimeError):
            owner.wait_ready()

    def test_force_cleanup(self):
        owner = ManagedSGLang('/weights', Settings())
        owner.process = MagicMock()
        owner.process.wait.side_effect = [subprocess.TimeoutExpired('sglang', 15), None]
        with patch('ntele_decision.launcher.os.killpg') as kill:
            owner.stop()
            self.assertEqual(kill.call_count, 2)
