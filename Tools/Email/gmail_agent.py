"""
C:/Users/amogh/Desktop/Project-Northlight/Tools/Email/gmail_agent.py
Gmail Email Agent - Complete implementation for sending and receiving emails
Prerequisites:
1. Enable Gmail API in Google Cloud Console
2. Create OAuth2 credentials (Desktop application)
3. Download credentials as 'credentials.json'
4. Install required libraries: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
"""

import os
import base64
import json
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pickle
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GmailAgent:
    """Gmail Agent for sending and receiving emails"""
    
    # If modifying these scopes, delete the file token.pickle.
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
    
    def __init__(self, credentials_file='credentials.json', token_file='token.pickle'):
        """Initialize Gmail Agent with authentication"""
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = self.authenticate()
        self.processed_messages = set()  # Track processed message IDs
        
    def authenticate(self):
        """Authenticate and return Gmail service object"""
        creds = None
        
        # Token file stores the user's access and refresh tokens
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        return build('gmail', 'v1', credentials=creds)
    
    # ==================== SENDING EMAILS ====================
    
    def send_email(self, to: str, subject: str, body: str, 
                   cc: Optional[List[str]] = None, 
                   bcc: Optional[List[str]] = None) -> Dict:
        """
        Send an email
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body (can be HTML)
            cc: List of CC recipients
            bcc: List of BCC recipients
            
        Returns:
            Dictionary with send status and message ID
        """
        try:
            message = MIMEMultipart()
            message['To'] = to
            message['Subject'] = subject
            
            if cc:
                message['Cc'] = ', '.join(cc)
            if bcc:
                message['Bcc'] = ', '.join(bcc)
            
            message.attach(MIMEText(body, 'html'))
            raw_message = base64.urlsafe_b64encode(
                message.as_bytes()).decode('utf-8')
            
            sent_message = self.service.users().messages().send(
                userId='me',
                body={'raw': raw_message}
            ).execute()
            
            print(f"Email sent successfully! Message ID: {sent_message['id']}")
            return {
                'success': True,
                'message_id': sent_message['id']
            }
            
        except HttpError as error:
            print(f"An error occurred: {error}")
            return {
                'success': False,
                'error': str(error)
            }
    

    # ==================== RECEIVING EMAILS ====================
    
    def get_messages(self, query: str = 'is:unread', max_results: int = 10) -> List[Dict]:
        """
        Get messages matching a query
        
        Args:
            query: Gmail search query (e.g., 'is:unread', 'from:someone@example.com')
            max_results: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        try:
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return []
            
            detailed_messages = []
            for msg in messages:
                msg_data = self.get_message_details(msg['id'])
                detailed_messages.append(msg_data)
            
            return detailed_messages
            
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []
    
    def get_message_details(self, message_id: str) -> Dict:
        """Get detailed information about a specific message"""
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=message_id
            ).execute()
            
            headers = message['payload'].get('headers', [])
            header_dict = {h['name']: h['value'] for h in headers}
            
            body = self._get_message_body(message['payload'])
            
            attachments = self._get_attachments_info(message['payload'])
            
            return {
                'id': message_id,
                'thread_id': message.get('threadId'),
                'from': header_dict.get('From', ''),
                'to': header_dict.get('To', ''),
                'subject': header_dict.get('Subject', ''),
                'date': header_dict.get('Date', ''),
                'body': body,
                'snippet': message.get('snippet', ''),
                'attachments': attachments,
                'labels': message.get('labelIds', [])
            }
            
        except HttpError as error:
            print(f"❌ Error getting message details: {error}")
            return {}
    
    def _get_message_body(self, payload: Dict) -> str:
        """Extract body from message payload"""
        body = ''
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8')
                elif part['mimeType'] == 'text/html':
                    data = part['body']['data']
                    body += base64.urlsafe_b64decode(data).decode('utf-8')
        elif payload['body'].get('data'):
            body = base64.urlsafe_b64decode(
                payload['body']['data']).decode('utf-8')
        
        return body
    
    def _get_attachments_info(self, payload: Dict) -> List[Dict]: # we will not need this
        """Get information about attachments"""
        attachments = []
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('filename'):
                    attachments.append({
                        'filename': part['filename'],
                        'mimeType': part.get('mimeType', ''),
                        'size': part['body'].get('size', 0),
                        'attachment_id': part['body'].get('attachmentId', '')
                    })
        
        return attachments
    
    def mark_as_read(self, message_id: str): 
        """Mark a message as read"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            print(f"✅ Message {message_id} marked as read")
        except HttpError as error:
            print(f"❌ Error marking message as read: {error}")
    
    # ==================== MONITORING EMAILS ====================
    
    def monitor_inbox(self, check_interval: int = 30, 
                     callback=None, query: str = 'is:unread'):
        """
        Monitor inbox for new messages (polling method)
        
        Args:
            check_interval: Seconds between checks
            callback: Function to call when new message arrives
            query: Gmail search query for filtering messages
        """
        print(f" Starting email monitoring (checking every {check_interval} seconds)...")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                messages = self.get_messages(query=query)
                
                for msg in messages:
                    msg_id = msg['id']
                    
                    # Check if we've already processed this message
                    if msg_id not in self.processed_messages:
                        self.processed_messages.add(msg_id)
                        
                        print(f"\n📨 New email received!")
                        print(f"From: {msg['from']}")
                        print(f"Subject: {msg['subject']}")
                        print(f"Preview: {msg['snippet'][:100]}...")
                        
                        # Call callback function if provided
                        if callback:
                            callback(msg)
                        
                        # Optional: Mark as read after processing
                        # self.mark_as_read(msg_id)
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n Monitoring stopped by user")
        except Exception as e:
            print(f"\n Error during monitoring: {e}")
    
    def get_recent_emails(self, minutes_ago: int = 30) -> List[Dict]: # Probably make this an agent on its own
        """Get emails received in the last X minutes"""
        # Gmail query for recent emails
        after_timestamp = int((datetime.now() - timedelta(minutes=minutes_ago)).timestamp())
        query = f'after:{after_timestamp}'
        
        return self.get_messages(query=query)
    
    def reply_to_email(self, message_id: str, reply_body: str) -> Dict:
        """Reply to an email thread"""
        try:
            # Get original message
            original = self.get_message_details(message_id)
            
            # Create reply
            message = MIMEText(reply_body, 'html')
            message['To'] = original['from']
            message['Subject'] = f"Re: {original['subject']}"
            message['In-Reply-To'] = message_id
            message['References'] = message_id
            
            # Encode and send
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            
            sent = self.service.users().messages().send(
                userId='me',
                body={'raw': raw, 'threadId': original['thread_id']}
            ).execute()
            
            print(f"✅ Reply sent! Message ID: {sent['id']}")
            return {'success': True, 'message_id': sent['id']}
            
        except Exception as e:
            print(f"❌ Error sending reply: {e}")
            return {'success': False, 'error': str(e)}


# ==================== EXAMPLE USAGE ====================

def example_callback(message: Dict):
    """Example callback function for processing new emails"""
    print("\n🔔 Processing new email with custom callback...")
    
    # You can add your custom logic here
    # For example: save to database, trigger webhook, send auto-reply, etc.
    
    # Example: Auto-reply to specific senders
    if 'important' in message['subject'].lower():
        print("📌 This seems important! Flagging for immediate attention...")


def main():
    """Example usage of the Gmail Agent"""
    
    print("🚀 Initializing Gmail Agent...")
    agent = GmailAgent()
    
    # Example 1: Send an email
    print("\n📤 SENDING EMAIL EXAMPLE:")
    result = agent.send_email(
        to='amogh@outlook.com',
        subject='Test Email from Gmail Agent',
        body='''
        <h2>Hello!</h2>
        <p>This is a test email sent from the Gmail Agent.</p>
        <p>It supports <b>HTML formatting</b> and attachments!</p>
        ''',
        # Optional: add CC, BCC, and attachments
        # cc=['cc@example.com']
    )
    
    # Example 2: Check unread emails
    print("\n📥 CHECKING UNREAD EMAILS:")
    unread_messages = agent.get_messages(query='is:unread', max_results=5)
    
    if unread_messages:
        for msg in unread_messages:
            print(f"\n📧 From: {msg['from']}")
            print(f"   Subject: {msg['subject']}")
            print(f"   Preview: {msg['snippet'][:100]}...")
    else:
        print("No unread messages found.")
    
    # Example 3: Monitor inbox for new emails
    print("\n👀 MONITORING INBOX:")
    print("Choose monitoring option:")
    print("1. Start continuous monitoring")
    print("2. Check recent emails (last 30 minutes)")
    print("3. Skip monitoring")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == '1':
        # Start monitoring (this will run continuously)
        agent.monitor_inbox(
            check_interval=30,  # Check every 30 seconds
            callback=example_callback,
            query='is:unread'
        )
    elif choice == '2':
        # Check recent emails
        recent = agent.get_recent_emails(minutes_ago=30)
        print(f"\nFound {len(recent)} emails from the last 30 minutes")
        for msg in recent:
            print(f"- {msg['from']}: {msg['subject']}")
    
    print("\n✨ Gmail Agent example completed!")


if __name__ == "__main__":
    main()