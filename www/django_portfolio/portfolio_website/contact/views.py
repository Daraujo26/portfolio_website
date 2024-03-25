from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .models import Contact
from django.core.mail import send_mail
import json

@csrf_exempt 
@require_POST
def submit_contact_form(request):
    data = json.loads(request.body)

    # Save submission to the database
    contact = Contact.objects.create(
        name=data.get('name'),
        email=data.get('email'),
        subject=data.get('subject'),
        message=data.get('message')
    )

    # Send an email notification
    send_mail(
        subject=f"New contact form submission from {contact.name}",
        message=contact.message,
        from_email=contact.email,
        recipient_list=['contact.davidaraujo@gmail.com'],
    )

    return JsonResponse({"success": "Your message has been sent successfully."})
