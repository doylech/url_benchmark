class NeptuneLogger:
    def __init__(self, run=None):
        self._run = run
        self.action_repeat = 1

    def log(self, name, data, step):
        if self._run:
            try:
                self._run[name].log(data, step=step.value * self.action_repeat)
            except:
                pass

    def video(self, name, data, step):
        if self._run:
            print("Saving video to neptune not implemented")

    def set_params(self, name, params):
        if self._run:
            self._run[name] = params
