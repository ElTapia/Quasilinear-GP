import sys
sys.path.append('../../../src/')

import json
import os
import re
import gzip

import numpy as np
from scipy.interpolate import CubicSpline

import QGP.Defocusing.Methods as Def
from Utilities import create_folder


def load_gz_npy(path):
    with gzip.GzipFile(path, 'r') as f:
        return np.load(f)


def parse_snapshot_index(file_name):
    match = re.match(r"w_(\d{10})\.npy\.gz$", file_name)
    if match is None:
        return None
    return int(match.group(1))


def cubic_resample_complex(x_old, w_old, x_new):
    w_old = np.asarray(w_old)
    real_spline = CubicSpline(x_old, np.real(w_old), bc_type='natural')
    imag_spline = CubicSpline(x_old, np.imag(w_old), bc_type='natural')
    return real_spline(x_new) + 1j * imag_spline(x_new)


def main():
    A = 0.638 #0.661
    sigma = 1 / 10
    L = np.pi
    pars = {"k": 1}

    N_old = 160_000
    Nt_old = 64_221 # 2 * 

    steps_before_threshold = 1000 #500
    threshold = np.sqrt(2) / 2

    refine_dx = False #True
    refine_dt = False #True

    dx_factor = 1 #10 #2**3 #2**2
    dt_factor = 1 #10 #2**2

    this_dir = os.path.dirname(os.path.abspath(__file__))
    previous_results_dir = os.path.abspath(
        os.path.join(
            this_dir,
            '../../Results/Blow_up/As/blow_up_N_160000_Nt_64221_sigma_0.1_second/'
            'blow_up_A_0.638',
            #'../../Results/Blow_up/As/Double_time/'
            #'blow_up_N_160000_Nt_128442_sigma_0.1_A_0.661',
        )
    )

    time = load_gz_npy(os.path.join(previous_results_dir, 'time.npy.gz'))
    norm_inf = load_gz_npy(os.path.join(previous_results_dir, 'norm_inf.npy.gz'))
    x_old = load_gz_npy(os.path.join(previous_results_dir, 'space.npy.gz'))

    valid_crossing_idx = np.where((norm_inf != 0) & (norm_inf < threshold))[0]
    if len(valid_crossing_idx) == 0:
        raise RuntimeError('No se encontro ningun instante con norm_inf < sqrt(2)/2.')

    idx_last_below = int(valid_crossing_idx[-1])
    T_star = float(time[idx_last_below])+ 10 * float(np.abs(time[1] - time[0]))

    target_idx = max(0, idx_last_below - steps_before_threshold)

    snapshot_indices = []
    for file_name in os.listdir(previous_results_dir):
        idx = parse_snapshot_index(file_name)
        if idx is not None:
            snapshot_indices.append(idx)

    if len(snapshot_indices) == 0:
        raise RuntimeError('No se encontraron archivos w_XXXXXXXXXX.npy.gz en la corrida previa.')

    snapshot_indices = sorted(snapshot_indices)
    candidates = [idx for idx in snapshot_indices if idx <= target_idx]
    if len(candidates) == 0:
        raise RuntimeError(
            'No existe snapshot guardado con indice <= indice objetivo de restart. '
            'Prueba reducir steps_before_threshold o guardar snapshots mas frecuentes.'
        )

    idx_restart = int(candidates[-1])
    t0 = float(time[idx_restart])

    if T_star <= t0:
        raise RuntimeError('El tiempo final T* no es mayor que el tiempo inicial del restart.')

    restart_file = os.path.join(previous_results_dir, f'w_{idx_restart:010d}.npy.gz')
    w0_old = load_gz_npy(restart_file)

    if refine_dx:
        if dx_factor < 1 or int(dx_factor) != dx_factor:
            raise ValueError('dx_factor debe ser un entero >= 1.')
        dx_factor = int(dx_factor)
        N_new = dx_factor * (N_old - 1) + 1
        x_new = np.linspace(-L, L, N_new)
        w0 = cubic_resample_complex(x_old, w0_old, x_new)
    else:
        N_new = N_old
        x_new = x_old
        w0 = w0_old

    dt_old = float(np.abs(time[1] - time[0]))
    if refine_dt:
        dt_new = dt_old / float(dt_factor)
    else:
        dt_new = dt_old

    Nt_new = int(np.ceil((T_star - t0) / dt_new)) + 1

    out_parent = os.path.join(previous_results_dir, 'restart_runs')
    create_folder(out_parent)

    refine_tag = []
    if refine_dx:
        refine_tag.append(f'dx_{dx_factor}x')
    if refine_dt:
        refine_tag.append(f'dt_{dt_factor}x')
    if not refine_tag:
        refine_tag.append('no_refine')

    out_folder_name = (
        f'restart_A_{A}_from_{idx_restart:010d}'
        f'_to_last_below_{idx_last_below:010d}'
        f'_minus_{steps_before_threshold}dt'
        f'_{"_".join(refine_tag)}'
        f'_Nt_{Nt_new}'
        f'_N_{N_new}'
    )
    out_path = os.path.join(out_parent, out_folder_name)
    create_folder(out_path)
    out_path = out_path + '/'

    metadata = {
        'A': A,
        'sigma': sigma,
        'L': L,
        'N_old': N_old,
        'N_new': N_new,
        'Nt_old': Nt_old,
        'steps_before_threshold': steps_before_threshold,
        'threshold': float(threshold),
        'idx_last_below_threshold': idx_last_below,
        'T_star': T_star,
        'target_idx': target_idx,
        'idx_restart_snapshot': idx_restart,
        't0_restart': t0,
        'dt_old': dt_old,
        'dt_new': dt_new,
        'Nt_new': Nt_new,
        'refine_dx': refine_dx,
        'refine_dt': refine_dt,
        'dx_factor': dx_factor,
        'dt_factor': dt_factor,
        'previous_results_dir': previous_results_dir,
        'restart_file': restart_file,
        'out_path': out_path,
    }
    with open(os.path.join(out_path, 'restart_info.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Running restart simulation for A=', A)
    print('Threshold sqrt(2)/2:', threshold)
    print('Last index below threshold:', idx_last_below, 'T*:', T_star)
    print('Requested restart target index:', target_idx)
    print('Selected saved restart index:', idx_restart, 't0:', t0)
    print('Old dt:', dt_old, 'New dt:', dt_new)
    print('Old N:', N_old, 'New N:', N_new)
    print('Nt_new:', Nt_new)
    print('Refine dx:', refine_dx, 'factor:', dx_factor)
    print('Refine dt:', refine_dt, 'factor:', dt_factor)
    print('Output folder:', out_path)

    save_every = max(1, Nt_new // 100)

    w_solver = Def.CNFD(
        L,
        N_new,
        Nt_new,
        T_star,
        pars,
        'neumann',
        ti=t0,
        path=out_path,
        save_as_files='w_',
        save_properties=True,
        order=2,
        background=0,
    )
    w_solver.set_init_array(w0)
    w_solver.solve(Nelem=save_every, norm_inf_stop=threshold)

    w0_saved = load_gz_npy(os.path.join(out_path, 'w_0000000000.npy.gz'))
    init_consistency_err = float(np.linalg.norm(w0_saved - w0, np.inf))

    time_new = load_gz_npy(os.path.join(out_path, 'time.npy.gz'))
    t_final_new = float(time_new[-1])
    tf_err = abs(t_final_new - T_star)

    norm_inf_new = load_gz_npy(os.path.join(out_path, 'norm_inf.npy.gz'))
    valid_new = norm_inf_new[norm_inf_new != 0]
    if len(valid_new) > 0:
        final_norm_new = float(valid_new[-1])
        below_count_new = int(np.sum(valid_new < threshold))
    else:
        final_norm_new = float('nan')
        below_count_new = 0

    print('\nValidation summary')
    print('init_consistency_inf_error:', init_consistency_err)
    print('t_final_new:', t_final_new, 'target_T*:', T_star, 'abs_error:', tf_err)
    print('final_norm_inf_new:', final_norm_new)
    print('count(norm_inf_new < sqrt(2)/2):', below_count_new)
    print('Done!')


if __name__ == '__main__':
    main()