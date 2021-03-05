import torch as t 


class CtcOutputProcess(t.nn.Module):
    def __init__(self, blank_id):
        super(CtcOutputProcess, self).__init__()
        self.blank_id = blank_id
        
    def _process_ctc(self, ctc_output):
        return t.argmax(ctc_output, -1)
    
    def _rebuild_ctc(self, ctc_linear_output_id):
        length = ctc_linear_output_id.size(1)
        former = t.nn.functional.pad(ctc_linear_output_id, (1, 0), value=self.blank_id)
        current = t.nn.functional.pad(ctc_linear_output_id, (0, 1), value=self.blank_id)
        ctc_linear_out_id = ctc_linear_output_id.masked_fill((current == former).narrow(1, 0, length), self.blank_id)
        return ctc_linear_out_id

    def forward(self, ctc_output):
        ctc_linear_output_id = self._process_ctc(ctc_output)
        ctc_linear_output_id = self._rebuild_ctc(ctc_linear_output_id)
        return ctc_linear_output_id


    
