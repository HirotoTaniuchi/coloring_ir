from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/NU/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='NU/test_-10/', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=100, help='how many test images to run')
        parser.add_argument('--skip_reals', action='store_true', help='real_A / real_B を保存しない')
        parser.add_argument('--skip_4ch', action='store_true', help='fake_4ch (RGBA合成) を生成・保存しない')
        self.isTrain = False
        return parser
