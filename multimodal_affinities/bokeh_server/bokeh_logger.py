from bokeh.models.widgets import Div
import logging


class BokehLogger:
    """ A logger class that plugs right into the gui.. """

    _console_border = '1px'
    _console_width = 1024
    _info_color = '#006400'
    _warning_color = 'yellow'
    _error_color = 'red'

    def __init__(self):
        text = '<table ' \
               'border=' + BokehLogger._console_border + \
               ' width=' + str(BokehLogger._console_width) + '><tr><td>' + \
               '<font color="#006400">' + \
               '<b>Status: </b>' + \
               'Session started (wait 7 seconds for gui to respond)..' + \
               '</font>' + \
               '</td></tr></table>'
        self.text_component = Div(text=text, width=BokehLogger._console_width)

    def _log_message(self, color, message):
        self.text_component.text = '<table ' \
                                   'border=' + BokehLogger._console_border + \
                                   ' width=' + str(BokehLogger._console_width) + '><tr><td>' + \
                                   '<font color="' + color + '">' + \
                                   '<b>Status: </b>' + \
                                   message + \
                                   '</font>' + \
                                   '</td></tr></table>'

    def info(self, msg):
        self._log_message(BokehLogger._info_color, msg)
        logging.info(msg)

    def warning(self, msg):
        self._log_message(BokehLogger._warning_color, msg)
        logging.warning(msg)

    def error(self, msg):
        self._log_message(BokehLogger._error_color, msg)
        logging.error(msg)

    def widget(self):
        return self.text_component
