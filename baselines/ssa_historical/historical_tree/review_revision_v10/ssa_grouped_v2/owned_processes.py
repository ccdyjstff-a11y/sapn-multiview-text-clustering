"""PID plus creation-time ownership checks, shared by runner and pure mocks."""


def identity_state(process, created):
    try:
        if not process.is_running():
            return 'gone', None
        return ('live', None) if process.create_time() == created else ('reused', None)
    except Exception as error:
        return 'unverified', repr(error)


def same_process(process, created):
    return identity_state(process, created)[0] == 'live'


def act_owned(entries, action):
    outcomes = []
    for (pid, created), process in reversed(list(entries.items())):
        state, error = identity_state(process, created)
        if state != 'live':
            outcomes.append(dict(pid=pid, create_time=created, action=action,
                                 result='error' if state == 'unverified' else 'identity_not_live_skipped',
                                 identity_state=state, error=error))
            continue
        try:
            getattr(process, action)()
            outcomes.append(dict(pid=pid, create_time=created, action=action, result='requested'))
        except Exception as error:
            outcomes.append(dict(pid=pid, create_time=created, action=action, result='error', error=repr(error)))
    return outcomes
