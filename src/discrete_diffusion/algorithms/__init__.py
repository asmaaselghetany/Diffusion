"""Algorithm implementations for discrete diffusion.

All OSS entrypoints resolve algorithms via Hydra `_target_` strings declared in
`configs/algo/*.yaml`. Example:

```
algo:
  _target_: discrete_diffusion.algorithms.mdlm.MDLM
  name: mdlm
  ...
```

See docs/01_algorithms.md for detailed documentation on algorithm structure.
"""

from .ar import AR  # noqa: F401
from .bd3lm import BD3LM  # noqa: F401
from .block_diffusion import BlockDiffusion  # noqa: F401
from .continuous_embedding_diffusion import ContinuousEmbeddingDiffusion  # noqa: F401
from .flexmdm_anyorder import FlexMDMAnyOrder  # noqa: F401
from .gidd import GIDD  # noqa: F401
from .jepa import LatentJEPATrainer  # noqa: F401
from .mdlm import MDLM  # noqa: F401
from .partition_mdlm import PartitionMDLM  # noqa: F401
from .sedd import SEDD  # noqa: F401
from .udlm import UDLM  # noqa: F401

__all__ = [
  'AR',
  'BD3LM',
  'BlockDiffusion',
  'ContinuousEmbeddingDiffusion',
  'FlexMDMAnyOrder',
  'GIDD',
  'LatentJEPATrainer',
  'MDLM',
  'PartitionMDLM',
  'SEDD',
  'UDLM',
]
