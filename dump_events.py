import os
import os.path as osp
import argparse
import glob
import csv

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def find_event_dir(path):
    """Accept an experiment dir, its events/ dir, or an event file directly."""
    if osp.isfile(path):
        return osp.dirname(path)
    if osp.isdir(osp.join(path, 'events')):
        return osp.join(path, 'events')
    return path


def derive_name(event_dir):
    """Build '<ss_method>_<teacher>_to_<student>' from the experiment dir name.

    Experiment dirs are named like
        '<ss_method>_student_<s_arch>_weight..._teacher_<t_arch>'
    where ss_method is e.g. sskd / sskd_rotation / sskd_jigsaw / sskd_exemplar.
    Falls back to the raw dir name if the pattern doesn't match.
    """
    exp_dir = event_dir
    if osp.basename(osp.normpath(event_dir)) == 'events':
        exp_dir = osp.dirname(osp.normpath(event_dir))
    exp_name = osp.basename(osp.normpath(exp_dir))

    if '_student_' not in exp_name or '_teacher_' not in exp_name:
        return exp_name
    ss_method, rest = exp_name.split('_student_', 1)
    s_arch = rest.split('_weight', 1)[0]
    t_arch = exp_name.split('_teacher_', 1)[1]
    return '{}_{}_to_{}'.format(ss_method, t_arch, s_arch)


def dump(event_dir, out_csv):
    ea = EventAccumulator(event_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if not tags:
        print('No scalar tags found in', event_dir)
        return

    # Collect into {step: {tag: value}} so each step is one wide row.
    rows = {}
    for tag in tags:
        for ev in ea.Scalars(tag):
            rows.setdefault(ev.step, {})[tag] = ev.value

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + tags)
        for step in sorted(rows):
            writer.writerow([step] + [rows[step].get(t, '') for t in tags])

    print('Wrote {} rows x {} tags -> {}'.format(len(rows), len(tags), out_csv))
    print('Tags:', ', '.join(tags))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export TensorBoard scalars to CSV.')
    parser.add_argument('path', help='experiment dir, events/ dir, or event file')
    parser.add_argument('-o', '--out', default=None, help='output csv path')
    args = parser.parse_args()

    event_dir = find_event_dir(args.path)
    out_csv = args.out or osp.join(event_dir, derive_name(event_dir) + '.csv')
    dump(event_dir, out_csv)
