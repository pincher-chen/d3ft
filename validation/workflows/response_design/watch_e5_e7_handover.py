"""CPU-only durable milestone journal; never submits or cancels GPU work."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

P = Path(__file__).resolve().parents[3]
Y = P / 'validation/results/response_design_20261015_e7_ct_pilot'
E = P / 'validation/results/response_design_20261016_e5_k4_transfer'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main(root, once=False):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statefile = root / 'observer.json'
    seen = set(read(statefile).get('seen', [])) if statefile.exists() else set()

    def event(key, message):
        key = hashlib.sha256(key.encode()).hexdigest()[:24]
        if key in seen:
            return
        now = dt.datetime.now().astimezone().isoformat()
        marker = '<!-- response_design-event:' + key + ' -->'
        # Check the document too: a crash after append but before state save
        # must not duplicate the scientific progress entry on restart.
        for dest in [P / 'response_design_to_do_list.md', P / 'paper/response_design_EXPERIMENTS_20261014.md']:
            if marker not in dest.read_text():
                with dest.open('a') as f:
                    f.write('\n- ' + now + ' ' + message + ' ' + marker + '\n')
                    f.flush()
                    os.fsync(f.fileno())
        with (root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=now, event_id=key, message=message), ensure_ascii=False) + '\n')
        seen.add(key)

    while True:
        try:
            indices = {'E5': read(E / 'active_transfer.json'), 'E7': read(Y / 'active_training.json')}
            for name, index in indices.items():
                r = Path(index['root'])
                if name == 'E7' and (r / 'yield_to_E5.json').exists():
                    event(str(r / 'yield_to_E5.json'), 'E7→E5自动交接请求已锁定：当前合格POINT及检查点保存后主动让卡；不取消计算。')
                for f in sorted(r.glob('*_launch.json')):
                    d = read(f)
                    status = d.get('status', 'unknown')
                    if name == 'E5':
                        event(str(f) + status, f'E5子步骤 {f.stem}：{status}，PID={d.get("pid")}，GPU={d.get("gpus")}；退出状态与物理验收分别记录。')
                    elif status == 'cooperative_paused':
                        event(str(f) + status, f'E7子步骤 {f.stem}：POINT与检查点已保存，exit75合作式暂停并释放GPU{d.get("gpus")}；不记物理失败。')
                if name == 'E7':
                    for points in (r / 'runs').glob('*/points.json'):
                        for row in read(points):
                            event(str(points) + row['label'],
                                  f'E7 {points.parent.name}/{row["label"]} POINT已落盘，'
                                  f'驻点残差={row.get("residual")}；完整材料验收另记。')
                    for history in r.glob('*_history.json'):
                        for row in read(history):
                            decision = row['acceptance']
                            event(str(history) + str(row['trial']), f'E7 {history.stem} trial{row["trial"]}：接受={decision.get("accepted")}，训练loss={decision.get("loss_after")}；留出不参与更新。')
                    for result in r.glob('*_result.json'):
                        if read(result).get('completed'):
                            event(str(result), 'E7拟合/折子步骤已完成并落盘：' + result.name)
                else:
                    for point in (r / 'runs').glob('*/point.json'):
                        if read(point).get('physical_passed'):
                            event(str(point), 'E5真实态物理门通过并落盘：' + point.parent.name)
                    for tag in ['start', 'endpoint']:
                        spectrum = r / (tag + '_k4_spectrum.json')
                        if spectrum.exists():
                            event(str(spectrum), f'E5 {tag}：64个k点谱数值验收已落盘，gap={read(spectrum)["fundamental_gap_eV"]}eV；尚需两态目标转移判定。')
                    cert = r / 'acceptance.json'
                    if cert.exists():
                        d = read(cert)
                        event(str(cert), f'E5两态k4完成：目标转移通过={d.get("target_transfer_passed")}，误差={d.get("target_error_meV")}meV；E7可按原门恢复。')
                    status = read(r / 'status.json')
                    if status.get('stage', '').startswith(('failed', 'infrastructure')):
                        event(str(r) + status['stage'], 'E5停止待审计：' + str(status) + '；未完成不称物理失败。')
            save(statefile, dict(time=dt.datetime.now().astimezone().isoformat(), pid=os.getpid(),
                seen=sorted(seen), controllers_alive={k: alive(v['controller_pid']) for k, v in indices.items()},
                GPU_submission_capability=False, interval_seconds=20))
        except (OSError, ValueError, KeyError) as exc:
            with (root / 'observer_errors.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=dt.datetime.now().astimezone().isoformat(), error=repr(exc))) + '\n')
        if once:
            return
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(args.root, args.once)
