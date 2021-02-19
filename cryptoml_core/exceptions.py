class NotFoundException(Exception):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__()

class MessageException(Exception):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__()