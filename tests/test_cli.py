import pytest
import torch

from kinema.cli import DEFAULTS, _apply_override, build, load_config, main, resolve_device

pytest.importorskip('yaml', reason = 'PyYAML is needed for config files')

def write_config(tmp_path, extra = ''):
    path = tmp_path / 'config.yaml'
    path.write_text(
        'model:\n'
        '  dim: 8\n'
        '  dim_mults: [1, 2]\n'
        '  attn_heads: 2\n'
        '  attn_dim_head: 8\n'
        'diffusion:\n'
        '  image_size: 16\n'
        '  num_frames: 2\n'
        '  timesteps: 10\n'
        '  sampling_timesteps: 3\n'
        + extra,
        encoding = 'utf-8'
    )
    return path


def test_defaults_load_without_a_config_file():
    config = load_config()
    assert config['diffusion']['image_size'] == DEFAULTS['diffusion']['image_size']


def test_config_file_merges_over_defaults(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config['model']['dim'] == 8
    # untouched sections survive the merge
    assert config['trainer']['results_folder'] == './results'


def test_load_config_does_not_mutate_defaults(tmp_path):
    load_config(write_config(tmp_path))
    assert DEFAULTS['model']['dim'] == 64


def test_set_overrides_are_typed():
    config = load_config(overrides = ['trainer.train_lr=3e-4', 'trainer.amp=true', 'data.folder=./clips'])
    assert config['trainer']['train_lr'] == pytest.approx(3e-4)
    assert config['trainer']['amp'] is True
    assert config['data']['folder'] == './clips'


def test_set_creates_missing_sections():
    config = load_config(overrides = ['extra.nested.key=1'])
    assert config['extra']['nested']['key'] == 1


def test_malformed_set_is_rejected():
    with pytest.raises(SystemExit):
        _apply_override({}, 'no-equals-sign')


def test_build_returns_a_working_model(tmp_path):
    diffusion = build(load_config(write_config(tmp_path)), torch.device('cpu'))
    assert diffusion.sample(batch_size = 1, progress = False).shape == (1, 3, 2, 16, 16)


def test_resolve_device_honours_an_explicit_name():
    assert resolve_device('cpu') == torch.device('cpu')


def test_train_then_sample_end_to_end(tmp_path, capsys):
    from kinema.data import video_tensor_to_gif

    clips = tmp_path / 'clips'
    clips.mkdir()
    for i in range(2):
        video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(clips / f'{i}.gif'))

    results = tmp_path / 'results'
    config = write_config(tmp_path)

    exit_code = main([
        'train', '-c', str(config), '--device', 'cpu',
        '--set', f'data.folder={clips}',
        '--set', f'trainer.results_folder={results}',
        '--set', 'trainer.train_num_steps=3',
        '--set', 'trainer.train_batch_size=1',
        '--set', 'trainer.save_and_sample_every=2',
        '--set', 'trainer.num_sample_rows=1',
        '--set', 'trainer.gradient_accumulate_every=1',
    ])
    assert exit_code == 0

    checkpoint = results / 'model-1.pt'
    assert checkpoint.exists(), 'training should have written a milestone checkpoint'

    out = tmp_path / 'out.gif'
    assert main([
        'sample', str(checkpoint), '-c', str(config),
        '--device', 'cpu', '-o', str(out), '--steps', '3', '-q'
    ]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert 'wrote' in capsys.readouterr().out


def test_sample_writes_one_file_per_video(tmp_path):
    from kinema.data import video_tensor_to_gif
    from kinema.trainer import Trainer

    clips = tmp_path / 'clips'
    clips.mkdir()
    video_tensor_to_gif(torch.rand(3, 2, 16, 16), str(clips / '0.gif'))

    config = write_config(tmp_path)
    diffusion = build(load_config(config), torch.device('cpu'))
    trainer = Trainer(diffusion, clips, train_batch_size = 1, results_folder = str(tmp_path / 'r'), device = 'cpu')
    trainer.save(0)

    out = tmp_path / 'many.gif'
    main(['sample', str(tmp_path / 'r' / 'model-0.pt'), '-c', str(config),
          '--device', 'cpu', '-o', str(out), '-n', '2', '--steps', '3', '-q'])

    assert (tmp_path / 'many-0.gif').exists()
    assert (tmp_path / 'many-1.gif').exists()


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        main(['--version'])
    assert excinfo.value.code == 0
