# -*- coding: utf-8 -*-
from django import forms


class DocumentForm(forms.Form):
    #docfile = forms.FileField()
    name = forms.CharField(max_length=100)
    docfile = forms.FileField()
