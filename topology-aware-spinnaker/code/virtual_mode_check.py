from pathlib import Path
import json

from spinn_machine.virtual_machine import virtual_machine
import spinnaker_graph_front_end as gfe
import pacman
import spinn_machine


def main():
    report = {}
    report['imports'] = {
        'spinnaker_graph_front_end': getattr(gfe, '__version__', 'unknown'),
        'pacman': getattr(pacman, '__version__', 'unknown'),
        'spinn_machine': getattr(spinn_machine, '__version__', 'unknown')
    }

    setup_ok = False
    vm_ok = False
    vm_error = None
    setup_error = None
    stop_ok = False
    try:
        gfe.setup(n_chips_required=1)
        setup_ok = True
        try:
            vm = virtual_machine(8, 8)
            ethernet_chips = list(vm.ethernet_connected_chips)
            report['virtual_machine'] = {
                'width': vm.width,
                'height': vm.height,
                'n_chips': vm.n_chips,
                'ethernet_connected_chips': len(ethernet_chips)
            }
            vm_ok = True
        except Exception as exc:
            vm_error = repr(exc)
        gfe.stop()
        stop_ok = True
    except Exception as exc:
        setup_error = repr(exc)
        try:
            gfe.stop()
        except Exception:
            pass

    report['frontend_probe'] = {
        'setup_ok': setup_ok,
        'stop_ok': stop_ok,
        'setup_error': setup_error,
        'virtual_machine_ok': vm_ok,
        'virtual_machine_error': vm_error
    }

    out = Path(__file__).with_name('virtual_mode_check_report.json')
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
