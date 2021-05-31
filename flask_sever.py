from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
from prediction import prediction_result_from_img, init_artificial_neural_network
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

# 程序开始时声明
sess = tf.Session()
graph = tf.get_default_graph()
model = init_artificial_neural_network(sess)
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST', 'GET'])
def getclass():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp,jpeg"})
        # user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        try:
            with graph.as_default():
                set_session(sess)
                print(upload_path)
                res = prediction_result_from_img(model, upload_path)
                resp = jsonify({'result': res})
                return resp
        except Exception as e:
            print('发生了异常-getclass：', e)
            return jsonify({"error": 1002, "msg": "'发生了异常-getclass" + e})
    return render_template('predict.html')


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
