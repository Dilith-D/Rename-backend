from django.urls import include,path
from . import views

urlpatterns = [
    path('object-detection/',views.object_detection_view, name='Object Detection'),
    
]
