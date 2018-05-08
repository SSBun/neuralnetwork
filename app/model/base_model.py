import json


class Response:
    error_code = 0
    log_msg = ''
    response = None

    def __init__(self, error_code=0, log_msg='', data=None):
        self.error_code = error_code
        self.log_msg = log_msg
        self.data = data

    def generate_to_json(self):
        jsonobject = {'error_code': self.error_code,
                      'log_msg': self.log_msg,
                      'data': self.data}
        return json.dumps(jsonobject)
