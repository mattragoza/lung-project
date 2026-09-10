# physics/warp/material

from . import forms


def _resolve_material_type(name: str):
    n = name.lower()
    if n in {'linear_elastic', 'linear', 'le'}:
        return LinearElasticMaterial
    elif n in {'st_venant_kirchoff', 'stvk', 'vk'}:
        return StVenantKirchoffMaterial
    elif n in {'neo_hookean', 'neohookean', 'nh'}:
        return NeoHookeanMaterial
    raise ValueError(f'Invalid material type: {name!r}')


class WarpMaterial:

    @staticmethod
    def get_linear():
        return LinearElasticMaterial

    @staticmethod
    def get_subclass(name: str):
        return _resolve_material_type(name)


class LinearElasticMaterial(WarpMaterial):
    residual_form = forms.linear_elastic_residual_form
    jacobian_form = forms.linear_elastic_jacobian_form
    is_linear = True


class StVenantKirchoffMaterial(WarpMaterial):
    residual_form = forms.st_venant_kirchoff_residual_form
    jacobian_form = forms.st_venant_kirchoff_jacobian_form
    is_linear = False


class NeoHookeanMaterial(WarpMaterial):
    residual_form = forms.neo_hookean_residual_form
    jacobian_form = forms.neo_hookean_jacobian_form
    is_linear = False

