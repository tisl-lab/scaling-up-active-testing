import glob
import os
import torch
import numpy as np

RESOLVE_LATEST = False  # set to False to always load base files (no _vN resolution)


def _resolve_latest(filename, ext):
    """Return filename (no ext) for the highest-version _vN variant, or base if none."""
    if not RESOLVE_LATEST:
        return filename
    pattern = f'{filename}_v*.{ext}'
    versions = glob.glob(pattern)
    if not versions:
        return filename
    def _ver(p):
        try:
            return int(os.path.splitext(p)[0].rsplit('_v', 1)[1])
        except (IndexError, ValueError):
            return -1
    return max(versions, key=_ver).removesuffix(f'.{ext}')

def save_tensor_(result, filename, i=0, add_duplicates=False):
    if add_duplicates:
        v_format = f'_v{i}' if i > 0 else ''
        if os.path.isfile(f'{filename}{v_format}.pt'):
            save_tensor_(result.cpu(), f'{filename}', i=i+1, add_duplicates=True)
        else:
            if i == 0:
                torch.save(result.cpu(), f'{filename}.pt')
            else:
                print(f'Warning: Saved as version {i}')
                torch.save(result.cpu(), f'{filename}_v{i}.pt')
    else:
        assert not(os.path.isfile(f'{filename}.pt')), 'This file already exists.'
        torch.save(result.cpu(), f'{filename}.pt')

def save_array_(result, filename, i=0, add_duplicates=False):
    if add_duplicates:
        v_format = f'_v{i}' if i > 0 else ''
        if os.path.isfile(f'{filename}{v_format}.npy'):
            save_array_(result, f'{filename}', i=i+1, add_duplicates=True)
        else:
            if i == 0:
                np.save(f'{filename}.npy', result)
            else:
                print(f'Warning: Saved as version {i}')
                np.save(f'{filename}_v{i}.npy', result)
    else:
        assert not(os.path.isfile(f'{filename}.npy')), 'This file already exists.'
        np.save(f'{filename}.npy', result)

def save_tensors(results, filenames, add_duplicates=False):
    if isinstance(results, list):
        for result, filename in zip(results, filenames):
            save_tensor_(result, filename, 0, add_duplicates)
    else:
        save_tensor_(results, filenames, 0, add_duplicates)

def load_tensors(filenames):
    if isinstance(filenames, list):
        results = []
        for filename in filenames:
            results.append(torch.load(f'{_resolve_latest(filename, "pt")}.pt', weights_only=True))
        return tuple(results)
    else:
        return torch.load(f'{_resolve_latest(filenames, "pt")}.pt', weights_only=True)

def save_arrays(results, filenames, add_duplicates=False):
    if isinstance(results, list):
        for result, filename in zip(results, filenames):
            save_array_(result, filename, 0, add_duplicates)
    else:
        save_array_(results, filenames, 0, add_duplicates)

def load_arrays(filenames):
    if isinstance(filenames, list):
        results = []
        for filename in filenames:
            results.append(np.load(f'{_resolve_latest(filename, "npy")}.npy'))
        return tuple(results)
    else:
        return np.load(f'{_resolve_latest(filenames, "npy")}.npy')

def clear_files(filenames):
    if isinstance(filenames, list):
        for filename in filenames:
            assert os.path.isfile(filename), 'This file does not exist.'
            os.remove(filename)
