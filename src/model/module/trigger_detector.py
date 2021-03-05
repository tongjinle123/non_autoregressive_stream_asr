import torch as t


class TriggerDetector:
    """
    single sample trigger detector

    """
    def __init__(self, blank_id=5):
        self.blank_id = blank_id
        self.last_id = None

    def _get_current_id(self, output_id):
        return output_id.squeeze(0)[-1].item()

    def _detect_policy(self, current_id):
        if current_id is None:
            return False

        if (current_id != self.blank_id) and (current_id != self.last_id):
            self.last_id = current_id
            return True
        else:
            self.last_id = current_id
            return False

    def detect(self, output_id):
        current_id = self._get_current_id(output_id)
        return self._detect_policy(current_id)


class DelayedTriggerDetector(TriggerDetector):
    def __init__(self, blank_id=5, delay=3):
        super(DelayedTriggerDetector, self).__init__(blank_id=blank_id)
        self.delay = delay

    def _get_current_id(self, output_id):
        try:
            return output_id.squeeze(0)[-(self.delay+1)].item()
        except:
            return None


class FullTriggerDetector(TriggerDetector):
    def __init__(self, blank_id=5):
        super(FullTriggerDetector, self).__init__(blank_id=blank_id)
        self.pre_triggered = False

    def detect(self, output_id):
        current_id = self._get_current_id(output_id)
        if_triggered = self._detect_policy(current_id)
        if self.pre_triggered and if_triggered:
            return True
        if if_triggered and not self.pre_triggered:
            self.pre_triggered = True
            return False
        else:
            return False


class FullTriggerDelayedDetector(FullTriggerDetector):
    def __init__(self, blank_id=5, delay=3):
        super(FullTriggerDelayedDetector, self).__init__(blank_id=blank_id)
        self.delay = delay

    def _get_current_id(self, output_id):
        try:
            return output_id.squeeze(0)[-(self.delay+1)].item()
        except:
            return output_id.squeeze(0)[0].item()

#
# class DelayedTriggerDetector:
#     """
#     delayed trigger
#     """
#     def __init__(self, blank_id=5, delay=3):
#         self.blank_id = blank_id
#         self.delay = delay
#         self.last_id = None
#         self.triggered = False
#         self.triggered_step = 0
#
#     def _sub_reset(self):
#         self.triggered = False
#         self.triggered_step = 0
#
#     def _get_current_id(self, output_id):
#         return output_id.squeeze(0)[-1].item()
#
#     def _hidden_detect(self, output_id):
#         current_id = self._get_current_id(output_id)
#         if (current_id != self.blank_id) and (current_id != self.last_id):
#             self.last_id = current_id
#             return True
#         else:
#             self.last_id = current_id
#             return False
#
#     def detect(self, output_id):
#         if self.triggered == False:
#             current_triggered = self._hidden_detect(output_id)
#             if current_triggered:
#                 self.triggered = True
#                 self.triggered_step += 1
#                 self.last_id = self._get_current_id(output_id)
#                 return False
#             else:
#                 self.last_id = self._get_current_id(output_id)
#                 return False
#         else:
#             current_triggered = self._hidden_detect(output_id)
#             current_id = self._get_current_id(output_id)
#             if current_triggered:
#                 if current_id in [self.last_id, self.blank_id]:
#                     self.triggered_step += 1
#                     if self.triggered_step > self.delay:
#                         return True
#                     else:
#                         return False
#
#                 else:   # current_id == other id
#                     self.triggered_step = 0
#                     return True
#
#             else:
#                 pass
#
#
#
