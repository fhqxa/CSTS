from torchtools.tt.arg import _parse_opts
from torchtools.tt.utils import *
from torchtools.tt.layer import *
from torchtools.tt.logger import *
from torchtools.tt.stat import *
from torchtools.tt.trainer import *


__author__ = 'namju.kim@kakaobrain.com'

# if tt.arg.seed:
#     np.random.seed(tt.arg.seed)
#     torch.manual_seed(tt.arg.seed)
# global command line arguments
arg = _parse_opts()
