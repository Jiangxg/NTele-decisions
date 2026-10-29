import subprocess
import unittest
from unittest.mock import MagicMock, patch

from ntele_decision.core import Settings
from ntele_decision.launcher import ManagedSGLang


class LauncherTests(unittest.TestCase):
    def test_command_and_ownership(self):
        owner = ManagedSGLang('/weights', Settings(), gpus='2,3')
        self.assertEqual(owner.command[-2:], ['--tp', '2'])
        with patch('ntele_decision.launcher.importlib.metadata.version', return_value='0.5.21'), patch('ntele_decision.launcher.socket.socket'), patch('pathlib.Path.open'), patch('ntele_decision.launcher.subprocess.Popen') as popen, patch('ntele_decision.launcher.os.killpg') as kill:
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
