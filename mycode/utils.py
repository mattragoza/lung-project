import sys, inspect, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_fenics

# functions for converting between image-like arrays 
#   and vectors of dofs for a finite element basis

def image_to_dofs(image, V):
    '''
    Args:
        image: (batch_size, image_size) torch.Tensor
        V: (mesh_size, 1) fenics.FunctionSpace
    Returns:
        dofs: (batch_size, mesh_size) torch.Tensor
    '''
    batch_size, image_size = image.shape

    coords = V.tabulate_dof_coordinates()
    coords = np.c_[coords, coords*0]
    coords = torch.as_tensor(coords, dtype=image.dtype, device=image.device)
    
    mesh_size, n_dims = coords.shape
    coords = coords[None,None,:,:]

    return F.grid_sample(
        input=image.view(batch_size,1,1,image_size),
        grid=coords.expand(batch_size,1,mesh_size,n_dims) * 2 - 1,
        align_corners=True
    ).view(batch_size, mesh_size)


def dofs_to_image(dofs, V, image_size):
    '''
    Args:
        dofs: (batch_size, mesh_size) torch.Tensor
        V: (mesh_size, 1) fenics.FunctionSpace
        image_size: int
    Returns:
        image: (batch_size, image_size) torch.Tensor
    '''
    batch_size, mesh_size = dofs.shape
    
    coords = np.linspace(0, 1, image_size)

    image = np.zeros((batch_size, image_size))
    for i in range(batch_size):
        func = torch_fenics.numpy_fenics.numpy_to_fenics(
            dofs[i].detach().cpu().numpy(), fe.Function(V)
        )
        for j in range(image_size):
            func.eval(image[i,j,None], x[j,None])

    return torch.as_tensor(image, device=dofs.device)


def as_bool(s):
    if isinstance(s, str):
        s = s.lower()
        if s in {'true', 't', '1'}:
            return True
        elif s in {'false', 'f', '0'}:
            return False
        else:
            raise ValueError(f'{repr(s)} is not a valid bool string')
    else:
        return bool(s)


def generate_arg_parser(func):

    # get full argument specification
    argspec = inspect.getfullargspec(func)
    args = argspec.args or []
    defaults = argspec.defaults or ()
    undefined = object() # sentinel object
    n_undefined = len(args) - len(defaults)
    defaults = (undefined,) * n_undefined + defaults

    # auto-generate argument parser
    parser = argparse.ArgumentParser()
    for name, default in zip(argspec.args, defaults):
        type_ = argspec.annotations.get(name, None)

        if default is undefined: # positional argument
            parser.add_argument(name, type=type_)

        elif default is False and type_ in {bool, None}: # flag
            parser.add_argument(
                f'--{name}', default=False, type=as_bool, help=f'[{default}]'
            )
        else: # optional argument
            if type_ is None and default is not None:
                type_ = type(default)
            parser.add_argument(
                f'--{name}', default=default, type=type_, help=f'[{default}]'
            )

    return parser


# decorator for auto parsing arguments and calling the main function

def main(func):

    parent_frame = inspect.stack()[1].frame
    __name__ = parent_frame.f_locals.get('__name__')

    if __name__ == '__main__':

        # parse and display command line arguments
        parser = generate_arg_parser(func)
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func
