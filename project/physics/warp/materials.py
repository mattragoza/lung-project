# physics/warp/materials.py

import warp as wp
import warp.fem

from . import forms


def _resolve_material_type(name: str):
    key = name.lower().replace('_', '')

    if key in {'linearelastic', 'linear', 'le'}:
        return LinearElasticMaterial

    elif key in {'stvenantkirchhoff', 'stvk', 'vk'}:
        return StVenantKirchhoffMaterial

    elif key in {'neohookean', 'neoh', 'nh'}:
        return CiarletNeoHookeanMaterial

    elif key in {'decoupledneohookean', 'decoup', 'dn'}:
        return DecoupledNeoHookeanMaterial

    elif key in {'yeohhyperelastic', 'yeoh', 'yh'}:
        return YeohHyperElasticMaterial

    raise ValueError(f'Invalid material type: {name!r}')


def get_material(name, **kwargs):
    cls = WarpMaterial.get_subclass(name)
    return cls(**kwargs)


class WarpMaterial:
    is_linear = False

    @staticmethod
    def get_subclass(name: str):
        return _resolve_material_type(name)


class LinearElasticMaterial(WarpMaterial):
    is_linear = True

    stress_func = forms.linear_elastic_stress
    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(stress_func)


class StVenantKirchhoffMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.st_venant_kirchhoff_stress
    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(stress_func)


class CiarletNeoHookeanMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.ciarlet_neohookean_stress
    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(stress_func)


class DecoupledNeoHookeanMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.decoupled_neohookean_stress
    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(stress_func)


class YeohHyperElasticMaterial(WarpMaterial):
    is_linear = False

    stress_func = forms.yeoh_hyperelastic_stress
    residual_form = forms.build_residual_form(stress_func)
    jacobian_form = forms.build_jacobian_form(stress_func)


LE = LinearElasticMaterial
