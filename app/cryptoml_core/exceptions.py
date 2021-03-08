

class MessageException(Exception):
    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__()


class NotFoundException(MessageException):
    def __init__(self, message):
        super(NotFoundException, self).__init__(message)
