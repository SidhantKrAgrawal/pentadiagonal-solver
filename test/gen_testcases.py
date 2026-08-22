import numpy as np

np.random.seed(1)
import scipy.sparse as sp
import scipy.linalg as lg
import scipy.sparse.linalg as slg

PRECISION = 15
PRECISION_FLOAT = 6


def calc_result_x(ds, dl, d, du, dw, x, precision=0):
    # first two column in ds is 0
    # first one column in dl is 0
    # last one column in du is 0
    # last two column in dw is 0
    #  assert all([v == 0 for v in ds[:,0:2].flat])
    u = np.zeros(x.shape)
    if precision > 0:
        ds = ds.round(precision)
        dl = dl.round(precision)
        d = d.round(precision)
        du = du.round(precision)
        dw = dw.round(precision)
        x = x.round(precision)

    ds = np.roll(ds, -2, axis=1)  # ds is indexed 2 ... N-1, cols: 0 .. N-3
    dl = np.roll(dl, -1, axis=1)  # ds is indexed 1 ... N-1, cols: 0 .. N-2
    du = np.roll(du, 1, axis=1)  # ds is indexed 0 ... N-2, cols: 1 .. N-1
    dw = np.roll(dw, 2, axis=1)  # ds is indexed 0 ... N-3, cols: 2 .. N-1
    for i in range(d.shape[0]):
        coeff_banded = [dw[i], du[i], d[i], dl[i], ds[i]]
        u[i] = lg.solve_banded((2, 2), coeff_banded, x[i])
        ds_diag = ds[i, :-2]  # dl is indexed 2 ... N-1 + after roll indexing
        dl_diag = dl[i, :-1]  # ds is indexed 1 ... N-1 + after roll indexing
        d_diag = d[i, :]  # d  is indexed 0 ... N-1
        du_diag = du[i, 1:]  # du is indexed 0 ... N-2 + after roll indexing
        dw_diag = dw[i, 2:]  # dw is indexed 0 ... N-3 + after roll indexing
        coeff_matrix = sp.diags([ds_diag, dl_diag, d_diag, du_diag, dw_diag],
                                [-2, -1, 0, 1, 2])
        #  u[i] = slg.spsolve(coeff_matrix, x[i])
        print(f'{i+1}/{d.shape[0]}: Condition number:',
              np.linalg.cond(coeff_matrix.toarray(),
                             np.inf))  #,np.max(np.abs(u[i] - v)))
    return u


def write_testcase(fname, ds, dl, dd, du, dw, x, u, u_float, solvedim):
    with open(fname, mode='w') as f:
        # Number of dimensions and solving dimension
        f.write(f'{len(ds.shape)} {solvedim}\n')
        # Sizes in different dimensions
        f.write(' '.join([str(size) for size in reversed(ds.shape)]))
        f.write('\n')
        # ds matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in ds.flatten()]))
        f.write('\n')
        # dl matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in dl.flatten()]))
        f.write('\n')
        # dd matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in dd.flatten()]))
        f.write('\n')
        # du matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in du.flatten()]))
        f.write('\n')
        # dw matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in dw.flatten()]))
        f.write('\n')
        # x matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in x.flatten()]))
        f.write('\n')
        # u matrix
        f.write(' '.join([str(round(val, PRECISION)) for val in u.flatten()]))
        f.write('\n')
        # u matrix in float precision
        f.write(' '.join(
            [str(round(val, PRECISION_FLOAT)) for val in u_float.flatten()]))
        f.write('\n')


def gen_testcases(fname, shape):
    N = shape[-1]
    n_sys = int(np.prod(shape[:-1]))
    print(f'{type(N)} {type(n_sys)}')
    ds = -1 + np.random.rand(int(n_sys), int(N)) * 0.1
    dl = -1 + np.random.rand(n_sys, N) * 0.1
    dd = 6 + np.random.rand(n_sys, N)
    du = -1 + np.random.rand(n_sys, N) * 0.1
    dw = -1 + np.random.rand(n_sys, N) * 0.1
    ds[..., :2] = 0  # dl is indexed 2 ... N-1
    dl[..., :1] = 0  # ds is indexed 1 ... N-1
    du[..., -1:] = 0  # du is indexed 0 ... N-2
    dw[..., -2:] = 0  # dw is indexed 0 ... N-3

    x = np.random.rand(n_sys, N)
    print('double')
    u = calc_result_x(ds, dl, dd, du, dw, x, PRECISION)
    print('float')
    u_float = calc_result_x(ds, dl, dd, du, dw, x, precision=PRECISION_FLOAT)

    # get back the requested shape
    ds = ds.reshape(*shape)
    dl = dl.reshape(*shape)
    dd = dd.reshape(*shape)
    du = du.reshape(*shape)
    dw = dw.reshape(*shape)
    x = x.reshape(*shape)
    u = u.reshape(*shape)
    u_float = u_float.reshape(*shape)

    # write solution for every solvedim
    for solvedim in range(len(shape)):
        write_testcase(fname + f'_solve{solvedim}', ds, dl, dd, du, dw, x, u,
                       u_float, solvedim)
        ds = ds.transpose(np.r_[np.arange(1, len(shape)), 0])
        dl = dl.transpose(np.r_[np.arange(1, len(shape)), 0])
        dd = dd.transpose(np.r_[np.arange(1, len(shape)), 0])
        du = du.transpose(np.r_[np.arange(1, len(shape)), 0])
        dw = dw.transpose(np.r_[np.arange(1, len(shape)), 0])
        x = x.transpose(np.r_[np.arange(1, len(shape)), 0])
        u = u.transpose(np.r_[np.arange(1, len(shape)), 0])
        u_float = u_float.transpose(np.r_[np.arange(1, len(shape)), 0])


def main():
    gen_testcases('one_dim_small', [5])
    gen_testcases('one_dim_large', [200])
    gen_testcases('two_dim_small', [8, 8])
    gen_testcases('two_dim_large', [32, 32])
    gen_testcases('three_dim_large', [32, 32, 32])
    #  gen_testcases('four_dim_large', [32, 32, 32, 32])


if __name__ == "__main__":
    main()
