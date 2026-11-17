"""Pure three-phase state machine. No processes, data, model or wall-clock access."""
import math


class PhaseClock:
    def __init__(self, run_id, worker_limit, startup_limit=300, shutdown_limit=60, pause_limit=1800):
        self.run_id = run_id
        self.limits = dict(startup=startup_limit, worker=worker_limit, shutdown=shutdown_limit)
        self.pause_limit = pause_limit
        self.phase = 'startup'
        self.last = 0.0
        self.wall = dict.fromkeys(self.limits, 0.0)
        self.active = dict.fromkeys(self.limits, 0.0)
        self.paused = False
        self.pause_started = None
        self.markers = {}
        self.boundary_intervals = {}
        self.max_poll_gap = 0.0
        self.status = None
        self.startup_wall_upper = None

    def _marker(self, marker, phase, sequence):
        if not isinstance(marker, dict):
            raise ValueError('marker is not an object')
        if isinstance(marker.get('schema'), bool) or isinstance(marker.get('sequence'), bool):
            raise ValueError('boolean schema/sequence is invalid')
        for key, value in [('schema', 1), ('run_id', self.run_id), ('phase', phase), ('sequence', sequence)]:
            if marker.get(key) != value:
                raise ValueError('marker identity/order mismatch: ' + key)
        if phase == 'finished':
            wall = marker.get('worker_call_wall_seconds')
            if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
                raise ValueError('invalid worker wall duration')
            if marker.get('outcome') not in ('returned', 'threw'):
                raise ValueError('invalid worker outcome')
        if phase in self.markers and self.markers[phase] != marker:
            raise ValueError('marker changed after observation')

    def observe(self, now, started=None, finished=None, exited=False, exit_code=None, marker_error=None):
        if self.status is not None:
            return self.status
        if not math.isfinite(now) or now < self.last:
            self.status = 'INVALID_MONOTONIC_CLOCK'
            return self.status
        previous = self.last
        dt = now - previous
        self.max_poll_gap = max(self.max_poll_gap, dt)
        self.wall[self.phase] += dt
        if not self.paused:
            self.active[self.phase] += dt
        self.last = now
        try:
            if marker_error:
                raise ValueError(marker_error)
            if finished is not None and started is None:
                raise ValueError('finished marker without started marker')
            if 'started' in self.markers and started is None:
                raise ValueError('started marker disappeared')
            if 'finished' in self.markers and finished is None:
                raise ValueError('finished marker disappeared')
            if started is not None:
                self._marker(started, 'started', 1)
                if 'started' not in self.markers:
                    if self.phase != 'startup':
                        raise ValueError('started marker in wrong phase')
                    self.markers['started'] = started
                    self.startup_wall_upper = now
                    self.boundary_intervals['started'] = [previous, now]
                    # Boundary could occur anywhere in this poll interval. Charge
                    # the full interval to worker, never giving unmetered worker time.
                    self.wall['startup'] -= dt
                    self.wall['worker'] += dt
                    if not self.paused:
                        self.active['startup'] -= dt
                        self.active['worker'] += dt
                    self.phase = 'worker'
            if finished is not None:
                self._marker(finished, 'finished', 2)
                if 'finished' not in self.markers:
                    if self.phase != 'worker':
                        raise ValueError('finished marker in wrong phase')
                    if finished['worker_call_wall_seconds'] > self.wall['worker'] + 0.01:
                        raise ValueError('worker marker duration exceeds observed worker wall upper bound')
                    self.markers['finished'] = finished
                    self.boundary_intervals['finished'] = [previous, now]
                    self.phase = 'shutdown'
        except ValueError as error:
            self.status = 'INVALID_PHASE_MARKER'
            self.marker_error = str(error)
            return self.status
        if self.paused and now - self.pause_started >= self.pause_limit:
            self.status = 'RESOURCE_SUSPEND_WAIT_CAP'
        elif (self.startup_wall_upper if self.startup_wall_upper is not None else now) >= self.limits['startup']:
            self.status = 'STARTUP_WALL_TIME_CAP'
        elif self.active['worker'] >= self.limits['worker']:
            self.status = 'WORKER_ACTIVE_TIME_CAP'
        elif ('finished' in self.boundary_intervals and
              now - self.boundary_intervals['finished'][0] >= self.limits['shutdown']):
            self.status = 'SHUTDOWN_WALL_TIME_CAP'
        elif exited:
            if exit_code != 0:
                self.status = 'PROCESS_NONZERO_EXIT'
            elif set(self.markers) != {'started', 'finished'}:
                self.status = 'MISSING_PHASE_MARKER_ON_EXIT'
            elif self.markers['finished']['outcome'] != 'returned':
                self.status = 'WORKER_EXCEPTION'
            else:
                self.status = 'COMPLETE_ZERO_EXIT_VALID_MARKERS'
        return self.status

    def set_paused(self, now, paused):
        assert now == self.last
        if paused and not self.paused:
            self.pause_started = now
        elif not paused:
            self.pause_started = None
        self.paused = paused

    def report(self):
        widths = {key: end - start for key, (start, end) in self.boundary_intervals.items()}
        return dict(phase=self.phase, status=self.status, phase_wall_seconds=self.wall.copy(),
                    phase_active_seconds=self.active.copy(), marker_boundary_intervals=self.boundary_intervals.copy(),
                    maximum_observed_poll_gap_seconds=self.max_poll_gap,
                    worker_boundary_overcharge_upper_seconds=sum(widths.values()),
                    startup_wall_upper_seconds=self.startup_wall_upper if self.startup_wall_upper is not None else self.last,
                    shutdown_wall_upper_seconds=(self.last - self.boundary_intervals['finished'][0]) if 'finished' in self.boundary_intervals else 0,
                    polling_error='Worker time conservatively includes both observed boundary intervals; this is whole-worker time, not pure PCBG optimization time.',
                    marker_error=getattr(self, 'marker_error', None))
