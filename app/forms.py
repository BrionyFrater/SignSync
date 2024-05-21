from flask_wtf import FlaskForm
from wtforms import SelectField
from flask_wtf.file import FileField, FileRequired, FileAllowed


class SearchForm(FlaskForm):
    select_field = SelectField('Search Signs')

class UploadForm(FlaskForm):
    video = FileField('Upload Video', validators=[FileRequired(), FileAllowed(['mov'], 'Only .mov files are allowed')])