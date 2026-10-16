"""Small OpenAI HTTP client. Chunk timing is not a per-token TPOT distribution."""
import json
import time
import urllib.request


class IncompleteResponse(RuntimeError):pass


def get(url,timeout=5):
    with urllib.request.urlopen(url,timeout=timeout) as response:return response.read().decode()


def post(url,body,timeout=120,cancel=None):
    request=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    issue=time.monotonic_ns();first=None;parts=[];ids=[];usage=None;reason=None;completed=False
    with urllib.request.urlopen(request,timeout=timeout) as response:
        if not body.get('stream'):
            result=json.loads(response.read());done=time.monotonic_ns()
            c=result['choices'][0];text=c.get('text',c.get('message',{}).get('content',''))
            return dict(text=text,tokens=c.get('token_ids'),usage=result.get('usage'),finish_reason=c.get('finish_reason'),
                        issue_ns=issue,first_ns=None,done_ns=done,request_id=result.get('id'))
        request_id=None
        for line in response:
            if cancel is not None and cancel.is_set():raise IncompleteResponse('client cancellation; server abort confirmation is not implied')
            if (time.monotonic_ns()-issue)/1e9>timeout:raise TimeoutError('total response deadline')
            if not line.startswith(b'data: '):continue
            payload=line[6:].strip()
            if payload==b'[DONE]':completed=True;break
            event=json.loads(payload)
            if event.get('error'):raise IncompleteResponse(str(event['error']))
            request_id=event.get('id',request_id)
            if event.get('usage') is not None:usage=event['usage']
            for choice in event.get('choices',[]):
                text=choice.get('text',choice.get('delta',{}).get('content')) or ''
                if text and first is None:first=time.monotonic_ns()
                parts.append(text)
                ids.extend(choice.get('token_ids') or [])
                if choice.get('finish_reason') is not None:reason=choice['finish_reason']
        if not completed or reason is None or not ''.join(parts):raise IncompleteResponse('stream ended without complete nonempty response')
    return dict(text=''.join(parts),tokens=ids or None,usage=usage,finish_reason=reason,
                issue_ns=issue,first_ns=first,done_ns=time.monotonic_ns(),request_id=request_id)
