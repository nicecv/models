# -*- coding: utf-8 -*-
import os
import time
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import urllib

parser = optparse.OptionParser()
parser.add_option(
    '-p', '--port',
    help="which port to serve content on",
    type='int', default=5555)

parser.add_option(
    '-t',
    '--template_file',
    help="the path of template file",
    type='str',
    default=
    '/raid/data/nice/metric_data/checked_logs/0004/info/index.html'
)
parser.add_option(
    '-o',
    '--output_anno_dir',
    help="the path of template file",
    type='str',
    default=
    '/raid/data/nice/metric_data/checked_logs/0005/anno'
)
    
opts, args = parser.parse_args()

template_folder, template_file_name = os.path.split(opts.template_file) 
# Obtain the flask app object
app = flask.Flask(__name__, template_folder=template_folder)

@app.route('/')
def index():
    return flask.render_template(template_file_name, has_result=False)

@app.route('/checked_names', methods=['POST'])
def checked_names():
    names = flask.request.form['names'].split(',')
    print names
    if len(names) > 1:    
        with open(os.path.join(opts.output_anno_dir, names[0].encode('utf-8')), 'w') as fw:
            fw.writelines([name.encode('utf-8')+'\n' for name in names[1:]])
    return flask.render_template(template_file_name, has_result=False)
    

def start_tornado(app, port=5555):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(opts.output_anno_dir):
        os.makedirs(opts.output_anno_dir)
    start_from_terminal(app)