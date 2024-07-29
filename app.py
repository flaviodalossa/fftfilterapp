from flask import Flask, request, send_file
from flask_cors import CORS
import os
import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
import tempfile
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

def fftfilter(input_path, param1, param2, param3, param4):
    logging.info("Starting fftfilter function")
    logging.info(f"Parameters: {param1}, {param2}, {param3}, {param4}")
    
    # Load the CSV file
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise
    
    # Perform FFT filter processing
    try:
        y = df['y'].values
        lower = param1 
        upper = param2
        fs = param3
        intercept = param4

        Fft_smooth = fft(y)
        Fft_smooth_p = np.copy(Fft_smooth)
        if intercept == "Yes":
            first = Fft_smooth[0]

        N = len(y)
        far = (fs / N) * np.arange(N)

        if upper is not None:
            indices2 = np.where(far <= upper)[0]
            index2 = indices2[-1]
            Fft_smooth[(index2+1):(N-index2)] = 0
        if lower is not None:
            indices1 = np.where(far >= lower)[0]
            index1 = indices1[0]
            Fft_smooth[:index1] = 0
            Fft_smooth[(N-index1+1):] = 0
        
        if intercept == "Yes":
            Fft_smooth[0] = first
        if intercept == "Not":
            Fft_smooth[0] = 0
        
        y_cut = np.real(ifft(Fft_smooth))
        
        df = pd.DataFrame({
            'FFt_filtered_i': Fft_smooth,
            'y_cut': y_cut,
        })

        output_path = input_path.replace('.csv', '_processed.csv')
        df.to_csv(output_path, index=False)
        logging.info(f"Processed file saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error during FFT processing: {e}")
        raise

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

        output_path = fftfilter(input_path, param1, param2, param3, param4)
        response = send_file(output_path, as_attachment=True, attachment_filename='processed.csv')

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
        return 'Internal Server Error', 500

if __name__ == '__main__':
    app.run(debug=True)
