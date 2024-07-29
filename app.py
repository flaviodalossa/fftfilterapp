# Importação das bibliotecas
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
import tempfile
import logging

# Configuração do app Flask e CORS
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

# Definição da função de processamento FFT
def fftfilter(input_path,param3, param4= 'Not', param1 =None, param2 = None):
    logging.info("Starting fftfilter function")
    logging.info(f"Parameters: {param1}, {param2}, {param3}, {param4}")

    try:
        df = pd.read_csv(input_path, skiprows=1)
        logging.info("CSV file read successfully")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return str(e), 400

    try:
        if df.shape[1] < 2:
            error_message = "O arquivo CSV deve conter pelo menos duas colunas."
            logging.error(error_message)
            return error_message, 400

        y = df.iloc[:, 1].values
        logging.info(f"Using second column as 'y': {y[:5]} (showing first 5)")
        
        try:
            lower = float(param1)
            upper = float(param2)
            fs = float(param3)
        except ValueError as e:
            error_message = "Certifique-se de que os parâmetros param1, param2 e param3 sejam números (float)."
            logging.error(error_message)
            return error_message, 400
    
        intercept = param4
        if intercept not in ["Yes", "Not"]:
            error_message = "O parâmetro param4 deve ser 'Yes' ou 'Not'."
            logging.error(error_message)
            return error_message, 400

        Fft_smooth = fft(y)
        first=Fft_smooth[0]
        N = len(y)
        far = (fs / N) * np.arange(N)

        if upper is not None:
            indices2 = np.where(far <= upper)[0]
            if len(indices2) > 0:
                index2 = indices2[-1]
                Fft_smooth[(index2+1):(N-index2)] = 0
                logging.info(f"Upper frequency limit applied at index {index2}")
            else:
                logging.warning("No indices found for upper frequency limit.")

        if lower is not None:
            indices1 = np.where(far >= lower)[0]
            if len(indices1) > 0:
                index1 = indices1[0]
                Fft_smooth[:index1] = 0
                Fft_smooth[(N-index1+1):] = 0
                logging.info(f"Lower frequency limit applied at index {index1}")
            else:
                logging.warning("No indices found for lower frequency limit.")

        if intercept == "Yes":
            Fft_smooth[0] = first
        elif intercept == "Not":
            Fft_smooth[0] = 0
        logging.info(f"Intercept adjusted: {intercept}")

        try:
            y_cut = np.real(ifft(Fft_smooth))
            logging.info("Inverse FFT applied successfully")
        except Exception as e:
            logging.error(f"Error during IFFT computation: {e}")
            return str(e), 500

        df_output = pd.DataFrame({
            'FFt_filtered_i': Fft_smooth,
            'y_cut': y_cut,
        })

        output_path = input_path.replace('.csv', '_processed.csv')
        df_output.to_csv(output_path, index=False)
        logging.info(f"Processed file saved to {output_path}")
        return output_path, 200
    except Exception as e:
        logging.error(f"Error during FFT processing: {e}")
        return str(e), 500

@app.route('/')
def upload_form():
    return '''
    <!doctype html>
    <title>Upload CSV</title>
    <h1>Upload CSV File</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
      <input type="file" name="csvfile">
      <p>Enter param1: <input type="text" name="param1"></p>
      <p>Enter param2: <input type="text" name="param2"></p>
      <p>Enter param3: <input type="text" name="param3"></p>
      <p>Enter param4 (Yes/Not): <input type="text" name="param4"></p>
      <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvfile' not in request.files:
        return 'No file part', 400
    file = request.files['csvfile']
    if file.filename == '':
        return 'No selected file', 400

    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, file.filename)
        file.save(input_path)
        logging.info(f"File saved to {input_path}")

        param1 = request.form.get('param1')
        param2 = request.form.get('param2')
        param3 = request.form.get('param3')
        param4 = request.form.get('param4')

        logging.info(f"Received parameters: param1={param1}, param2={param2}, param3={param3}, param4={param4}")
        logging.info(f"File path: {input_path}")

        output_path, status_code = fftfilter(input_path, param1, param2, param3, param4)
        if status_code != 200:
            return output_path, status_code

        response = send_file(output_path, as_attachment=True, download_name='processed.csv')

        @response.call_on_close
        def cleanup():
            try:
                os.remove(input_path)
                os.remove(output_path)
                os.rmdir(temp_dir)
                logging.info("Temporary files and directory cleaned up")
            except Exception as e:
                logging.error(f"Error cleaning up files: {e}")

        return response
    except Exception as e:
        logging.error(f"Error in /upload endpoint: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
