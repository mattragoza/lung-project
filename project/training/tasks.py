from typing import List, Dict

from ..core import utils


VALID_INPUTS  = {'image', 'material', 'mask'}
VALID_PHYSICS = {'E', 'nu', 'G', 'K', 'mu', 'lam', 'rho'}
VALID_TARGETS = {'image', 'material'} | VALID_PHYSICS
VALID_LOSSES  = {'ce', 'mse', 'msre', 'sim'}


class TaskSpec:

    def __init__(
        self,
        inputs: List[str],
        targets: List[str],
        losses: Dict[str, str],
        weights: Dict[str, float] = None,
        n_mat_labels: int = 5,
        rgb: bool = False
    ):
        self.inputs  = list(inputs)
        self.targets = list(targets)
        self.losses  = dict(losses)
        self.weights = dict(weights or {})

        self.n_mat_labels = n_mat_labels
        self.rgb = rgb

        self._validate()

    @property
    def valid_physics(self):
        return VALID_PHYSICS

    def _validate(self):

        utils.log(f'Inputs:  {self.inputs}')
        utils.log(f'Targets: {self.targets}')
        utils.log(f'Losses:  {self.losses}')

        assert len(self.inputs)  > 0
        assert len(self.targets) > 0
        assert len(self.losses)  > 0

        for input_ in self.inputs:
            if input_ not in VALID_INPUTS:
                raise ValueError(f'Invalid input: {input_}')

        for target in self.targets:
            if target not in VALID_TARGETS:
                raise ValueError(f'Invalid target: {target}')

        for target, loss in self.losses.items():
            loss_l = loss.lower()
            if target not in self.targets:
                raise ValueError(f'Invalid loss target: {target}')
            if loss_l not in self.valid_losses:
                raise ValueError(f'Invalid loss function: {loss}')
            if loss_l == 'sim' and target not in VALID_PHYSICS:
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
    def physics_outputs(self) -> List[str]:
        return [t for t in self.targets if t in VALID_PHYSICS]

    @property
    def has_physics_output(self) -> bool:
        return len(self.physics_outputs) > 0

    @property
    def has_physics_loss(self) -> bool:
        return any(l.lower() == 'sim' for l in self.losses.values())

    @property
    def image_channels(self) -> int:
        return 3 if self.rgb else 1

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

    def input_key(self, input_: str, visual: bool=False) -> str:
        if input_ == 'image':
            return 'img_true'
        elif input_ == 'material':
            return 'mat_true' if visual else 'mat_onehot'
        elif input_ == 'mask':
            return 'mask'
        raise ValueError(input_)

    def output_key(self, target: str, visual: bool=False) -> str:
        if target == 'image':
            return 'img_pred'
        elif target == 'material':
            return 'mat_pred' if visual else 'mat_logits'
        elif target in VALID_PHYSICS:
            return f'{target}_pred'
        raise ValueError(target)

    def target_key(self, target: str, visual: bool=False) -> str:
        if target == 'image':
            return 'img_true'
        elif target == 'material':
            return 'mat_true'
        elif target in VALID_PHYSICS:
            return f'{target}_true'
        raise ValueError(target)

    @property
    def metric_keys(self) -> List[str]:
        keys = ['loss', 'loss_base', 'loss_ratio', 'grad_norm']
        for tgt in self.targets:
            output_key = self.output_key(tgt)
            target_key = self.target_key(tgt)
            keys.append(output_key + '.mean')
            keys.append(output_key + '.std')
            keys.append(target_key + '.mean')
            keys.append(target_key + '.std')
        return keys 

    @property
    def viewer_keys(self) -> List[str]:
        keys = []
        for inp in self.inputs:
            input_key = self.input_key(inp, visual=True)
            keys.append(input_key)
        for tgt in self.targets:
            output_key = self.output_key(tgt, visual=True)
            target_key = self.target_key(tgt, visual=True)
            keys.extend([output_key, target_key])
        return keys

