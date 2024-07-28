from flask import Flask, request, send_file
from flask_cors import CORS
import os
import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
import tempfile

app = Flask(__name__)
CORS(app)

def fftfilter(input_path, param1, param2, param3, param4):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # Perform FFT filter processing
    y = df['y'].values
    lower = param1
    upper = param2
    fs = param3
    intercept = param4

    # Apply FFT in original data points
    Fft_smooth = fft(y)
    Fft_smooth_p = np.copy(Fft_smooth)
    if intercept == "Yes":
        first = Fft_smooth[0]  # it stores the first entry

    # Number of points
    N = len(y)

    # Frequency vector
    far = (fs / N) * np.arange(N)  # Hz

    # Shifted symmetric filtering
    if upper is not None:
        indices2 = np.where(far <= upper)[0]  # Setting indexes
        index2 = indices2[-1]
        Fft_smooth[(index2 + 1):(N - index2)] = 0
    if lower is not None:
        indices1 = np.where(far >= lower)[0]  # Setting indexes
        index1 = indices1[0]
        Fft_smooth[:index1] = 0
        Fft_smooth[(N - index1 + 1):] = 0
    # Restore the intercept
    if intercept == "Yes":
        Fft_smooth[0] = first
    if intercept == "Not":
        Fft_smooth[0] = 0
    # Inverse FFT
    y_cut = np.real(ifft(Fft_smooth))  # Get the real part

    df = pd.DataFrame({
        'FFt_filtered_i': Fft_smooth,
        'y_cut': y_cut,
    })

    # Save the processed CSV to a temporary file
    output_path = input_path.replace('.csv', '_processed.csv')
    df.to_csv(output_path, index=False)

    return output_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvfile' not in request.files:
        return 'No file part', 400
    file = request.files['csvfile']
    if file.filename == '':
        return 'No selected file', 400

    # Save file to a temporary location
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    file.save(input_path)

    # Get parameters
    param1 = request.form.get('param1')
    param2 = request.form.get('param2')
    param3 = request.form.get('param3')
    param4 = request.form.get('param4')

    # Process the CSV file using fftfilter
    output_path = fftfilter(input_path, param1, param2, param3, param4)

    # Send the processed file
    response = send_file(output_path, as_attachment=True, attachment_filename='processed.csv')

    # Cleanup files
    @response.call_on_close
    def cleanup():
        os.remove(input_path)
        os.remove(output_path)
        os.rmdir(temp_dir)

    return response

if __name__ == '__main__':
    app.run(debug=True)
