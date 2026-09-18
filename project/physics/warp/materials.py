# physics/warp/materials.py

import warp as wp
import warp.fem

from . import forms


def _resolve_material_type(name: str):
    key = name.lower().replace('_', '')

    if key in {'linearelastic', 'linear', 'le'}:
        return LinearElasticMaterial

    elif key in {'stvenantkirchoff', 'stvk', 'vk'}:
        return StVenantKirchoffMaterial

    elif key in {'neohookean', 'neoh', 'nh'}:
        return NeoHookeanMaterial

    elif key in {'yeohhyperelastic', 'yeoh', 'yh'}:
        return YeohHyperElasticMaterial

    raise ValueError(f'Invalid material type: {name!r}')


class WarpMaterial:
    is_linear = False

    @staticmethod
    def get_subclass(name: str):
        return _resolve_material_type(name)


class LinearElasticMaterial(WarpMaterial):
    is_linear = True

    stress_func = forms.linear_elastic_stress
    tangent_func = forms.linear_elastic_tangent

    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(tangent_func)


class StVenantKirchoffMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.st_venant_kirchoff_stress
    tangent_func = forms.st_venant_kirchoff_tangent

    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(tangent_func)


class NeoHookeanMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.neo_hookean_stress
    tangent_func = forms.neo_hookean_tangent

    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(tangent_func)


class YeohHyperElasticMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.yeoh_hyperelastic_stress
    tangent_func = forms.yeoh_hyperelastic_tangent

    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(tangent_func)

