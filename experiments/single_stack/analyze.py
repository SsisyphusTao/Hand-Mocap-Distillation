"""Analyze the single-stack 38-epoch experiment; optionally render phone replay."""
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate_completion_summary(complete, rows):
    """Bind summary claims to the epoch metrics independently checked by report()."""
    expected = dict(complete=True, epochs=38, updates_per_arm=282036,
                    images_per_arm=9025152, augmented_views_per_arm=18050304)
    for key, value in expected.items():
        if complete.get(key) != value:
            raise ValueError(f'Completion summary mismatch: {key}')
    if len(rows) != 38:
        raise ValueError('Expected 38 verified epochs')
    final = {a: rows[-1]['arms'][a]['metrics'] for a in ('A', 'B')}

    def compare(actual, wanted, path):
        if isinstance(wanted, dict):
            if not isinstance(actual, dict) or actual.keys() != wanted.keys():
                raise ValueError(f'Completion summary mismatch: {path}')
            for key in wanted:
                compare(actual[key], wanted[key], path + '.' + key)
        elif not isinstance(actual, (int, float)) or not math.isclose(actual, wanted, rel_tol=0, abs_tol=1e-4):
            raise ValueError(f'Completion summary mismatch: {path}')

    compare(complete.get('final'), final, 'final')
    best = {}
    for arm in ('A', 'B'):
        row = min(rows, key=lambda r: r['arms'][arm]['metrics']['mpjpe_mm'])
        best[arm] = dict(epoch=row['epoch'], metrics=row['arms'][arm]['metrics'])
        if complete.get('best_development_epoch', {}).get(arm) != row['epoch']:
            raise ValueError(f'Completion summary mismatch: best epoch {arm}')
    return final, best


def report(run, output, phone_baseline=None):
    import json
    import hashlib
    import shutil
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from hand_kd.utils import save_json, sha256
    from hand_kd.evaluation import summarize_errors
    from hand_kd.model import setup_torch

    RUN, OUT = Path(run).resolve(), Path(output).resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    complete=json.loads((RUN/'complete.json').read_text())
    config=json.loads((RUN/'config.json').read_text())
    assert complete['complete'] and complete['epochs']==38
    state=torch.load(RUN/'latest.pt',map_location='cpu',weights_only=False)
    assert state['next_epoch']==38 and state['global_batches']==282036
    for arm in ('A','B'):
        assert {int(v['step']) for v in state['arms'][arm]['optimizer']['state'].values()}=={282036}
        saved=torch.load(RUN/f'epoch-037/{arm}-student.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,saved[k]) for k,v in state['arms'][arm]['model'].items())
        del saved
    rows=[];predictions={};history_checks=[]
    for i in range(38):
        row=json.loads((RUN/f'epoch-{i:03d}/result.json').read_text())
        assert row['epoch']==i+1 and row['updates_per_arm']==(i+1)*7422 and row['teacher_unchanged']
        for arm in ('A','B'):
            p=np.load(RUN/f'epoch-{i:03d}/{arm}.npz')
            metrics=summarize_errors(p['prediction'],p['truth'])
            for key in ('mpjpe_mm','pa_mpjpe_mm','p95_joint_error_mm'):
                assert abs(metrics[key]-row['arms'][arm]['metrics'][key])<1e-4
            assert sha256(RUN/f'epoch-{i:03d}/{arm}-student.pt')==row['arms'][arm]['checkpoint_sha256']
            if i==37:predictions[arm]={k:p[k] for k in p.files}
        rows.append(row)
        history_checks.append(dict(epoch=i+1,prediction_metrics_and_checkpoint_hashes_verified=True))
    final, best = validate_completion_summary(complete, rows)
    np.testing.assert_array_equal(predictions['A']['ids'],predictions['B']['ids'])
    np.testing.assert_array_equal(predictions['A']['truth'],predictions['B']['truth'])
    teacher=np.load(RUN/'teacher-validation.npz')
    np.testing.assert_array_equal(teacher['ids'],predictions['A']['ids'])
    np.testing.assert_array_equal(teacher['truth'],predictions['A']['truth'])
    teacher_metrics=summarize_errors(teacher['prediction'],teacher['truth'])
    assert abs(teacher_metrics['mpjpe_mm']-complete['teacher']['mpjpe_mm'])<1e-4
    chain='';count=0
    for name in state['logs']:
        with (RUN/name).open() as stream:
            for line in stream:
                r=json.loads(line);count+=1
                assert r['global_batch']==count
                epoch=(count-1)//7422;batch=(count-1)%7422+1
                assert r['epoch']==epoch+1 and r['batch']==batch
                assert r['effective_loss_epoch']==epoch
                assert abs(r['lr']-(.001 if epoch<30 else .0001))<1e-12
                assert all(np.isfinite(list(r['arms'][a].values())).all() for a in ('A','B'))
                chain=hashlib.sha256((chain+r['batch_sha256']+r['teacher_feature_sha256']).encode()).hexdigest()
    assert count==282036 and chain==state['batch_chain']==rows[-1]['batch_chain']
    errors={}
    for arm,p in predictions.items():
        j,g=p['prediction'].astype(float),p['truth'].astype(float)
        errors[arm]=np.linalg.norm((j-j[:,:1])-(g-g[:,:1]),axis=-1).mean(1)*1000
    delta=errors['B']-errors['A'];rng=np.random.default_rng(0)
    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    result=dict(final=final,teacher=teacher_metrics,delta_B_minus_A_mm=float(delta.mean()),
        mpjpe_reduction_percent=float(100*(1-errors['B'].mean()/errors['A'].mean())),
        paired_image_bootstrap_95_interval_mm=np.percentile(boot,[2.5,97.5]).tolist(),
        B_better_image_fraction=float((delta<0).mean()),
        best_development=best,
        verified_epochs=history_checks,verified_updates_per_arm=count,shared_hash_chain_verified=True,
        final_checkpoint_matches_shared_state=True,
        limitations='One seed, reused 256-image development set. Image bootstrap is exploratory, not subject/session/seed uncertainty. Fixed final epoch38 is primary.')
    save_json(OUT/'metric-verification.json',result)
    fig,axes=plt.subplots(1,3,figsize=(15,4.3))
    for a,label in [('A','A: supervised'),('B','B: feature KD')]:
        for ax,key in zip(axes[:2],('mpjpe_mm','pa_mpjpe_mm')):
            ax.plot(range(1,39),[r['arms'][a]['metrics'][key] for r in rows],label=label)
        axes[2].plot(range(1,39),[r['arms'][a]['average_losses']['loss'] for r in rows],label=label)
    for ax,key,title in zip(axes[:2],('mpjpe_mm','pa_mpjpe_mm'),('Development MPJPE (mm)','Development PA-MPJPE (mm)')):
        ax.axhline(teacher_metrics[key],ls='--',color='gray',label='Frozen teacher');ax.set_ylabel(title)
    axes[2].set_ylabel('Native training loss (excludes KD)')
    for ax in axes:ax.set_xlabel('Completed epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Full data / 38 epochs / single-stack MobRecon: one seed, reused development set')
    fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=160);plt.close(fig)
    model_inputs=dict(arms={a:dict(final_checkpoint=str(RUN/f'epoch-037/{a}-student.pt')) for a in ('A','B')})
    save_json(OUT/'phone-inputs.json',model_inputs)
    save_json(OUT/'provenance.json',dict(run=str(RUN),config=config,checkpoint_sha256={a:sha256(Path(model_inputs['arms'][a]['final_checkpoint'])) for a in ('A','B')},renderer_sha256=sha256(ROOT/'experiments/single_stack/replay.py'),primary_epoch=38))
    shutil.copyfile(ROOT/'experiments/single_stack/replay.py',OUT/'replay-renderer-source.py')
    del state
    if phone_baseline is not None:
        from experiments.single_stack.replay import phone_replay
        setup_torch()
        phone_replay(RUN,OUT,model_inputs,baseline=phone_baseline,config=config)
    print(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed full38 run directory')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten')
    parser.add_argument('--phone-baseline', type=Path,
                        help='Optional verified teacher replay directory containing frames, detections and provenance')
    args = parser.parse_args()
    report(args.run, args.output, args.phone_baseline)


if __name__ == '__main__':
    main()
