This is my solution of the HW#1 of Ml in Production MADE course.

To run the model please install all the dependencies:
~~~
pip install -r requirements.txt
~~~

Then run the following for model training:
~~~
python -m src.train --config `path_to_cfg`
~~~
and run the following for predictions:
~~~
python -m src.predict --data `path_to_dataset` --model `path_to_model` --output `output_file_path`
~~~