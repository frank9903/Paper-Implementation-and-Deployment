# -*- coding: utf-8 -*-

from myproject.myapp.models import Document
from myproject.myapp.forms import DocumentForm
from django.http import HttpResponse
import json
from django.views.decorators.csrf import csrf_exempt
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/shuheng/Documents/iCloud/2019F/Paper-Implementation-and-Deployment/Mask RCNN/')
import run_inference

@csrf_exempt

def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            print("Image URL:" + os.getcwd()+newdoc.docfile.url)
            save_path=os.getcwd()+"/media/result.png"
            run_inference.run(os.getcwd()+newdoc.docfile.url, save_path=save_path)
            print("Result URL:" + save_path)
            
            # Redirect to the document list after POST
            return HttpResponse(json.dumps({"Status": 0,"ImgName": "/media/result.png"}, sort_keys=True))
        else:
            return HttpResponse(json.dumps({"Status": 1}, sort_keys=True))
