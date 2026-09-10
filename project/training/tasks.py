# training/tasks.py

from typing import List, Dict

from ..core import utils


VALID_INPUTS  = {'image', 'material', 'mask'}
VALID_PHYSICS = {'E', 'nu', 'G', 'K', 'mu', 'lam', 'rho'}
VALID_TARGETS = {'image', 'material'} | VALID_PHYSICS
VALID_LOSSES  = {'CE', 'MSE', 'MSRE', 'SIM'}


class TaskSpec:

    def __init__(
        self,
        inputs:  List[str],
        targets: List[str],
        losses:  Dict[str, str],
        weights: Dict[str, float] = None
    ):
        self.inputs  = list(inputs)
        self.targets = list(targets)
        self.losses  = dict(losses)
        self.weights = dict(weights or {})

        utils.log(f'Inputs:  {self.inputs}')
        utils.log(f'Targets: {self.targets}')
        utils.log(f'Losses:  {self.losses}')

        self._validate()

    def _validate(self):

        if len(self.inputs) < 1:
            raise ValueError('Task has no inputs')

        for input_ in self.inputs:
            if input_ not in VALID_INPUTS:
                raise ValueError(f'Invalid input: {input_}')

        if len(self.targets) < 1:
            raise ValueError('Task has no targets')

        for target in self.targets:
            if target not in VALID_TARGETS:
                raise ValueError(f'Invalid target: {target}')

        if len(self.losses) < 1:
            raise ValueError('Task has no losses')

        for target, loss in self.losses.items():
            loss_ = loss.upper()

            if target not in self.targets:
                raise ValueError(f'Invalid loss target: {target}')

            if loss_ not in VALID_LOSSES:
                raise ValueError(f'Invalid loss function: {loss}')

            if loss_ == 'sim' and target not in VALID_PHYSICS:
                raise ValueError(f'Invalid physics target: {target}')

        for target, weight in self.weights.items():

            if target not in self.losses:
                raise ValueError(f'Invalid weight target: {target}')

            if weight < 0.0:
                raise ValueError(f'Invalid weight value: {weight}')

    @property
    def loss_targets(self):
        return [t for t in self.targets if t in self.losses]

    @property
    def physics_targets(self) -> List[str]:
        return [t for t in self.targets if t in VALID_PHYSICS]

    @property
    def has_physics_target(self) -> bool:
        return len(self.physics_targets) > 0

    @property
    def has_physics_loss(self) -> bool:
        return any(l.lower() == 'sim' for l in self.losses.values())

    @property
    def image_channels(self) -> int:
        return 1

    @property
    def material_labels(self) -> int:
        return self.n_mat_labels

    @property
    def in_channels(self) -> int:
        total = 0
        for input_ in self.inputs:
            if input_ == 'image':
                total += self.image_channels
            elif input_ == 'material':
                total += self.material_labels + 1
            elif input_ == 'mask':
                total += 1
            else:
                raise ValueError(input_)
        return total

    def out_channels(self, target: str) -> int:
        if target == 'image':
            return self.image_channels
        elif target == 'material':
            return self.material_labels + 1
        elif target in VALID_PHYSICS:
            return 1
        raise ValueError(target)

    def metric_profile(self, target: str) -> str:
        if target in {'mask', 'material'}:
            return 'label'
        elif target in {'u'}:
            return 'vector'
        else:
            return 'scalar'

