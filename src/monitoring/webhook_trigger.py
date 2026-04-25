"""
Receives Grafana alert webhooks and triggers GitHub Actions.
Run this alongside your API.
"""
from fastapi import FastAPI, Request
import httpx, os, json

app = FastAPI(title="Alert Webhook")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "your_token_here")
GITHUB_REPO  = os.environ.get("GITHUB_REPO",  "YOURUSERNAME/fraud-detection")

@app.post("/webhook/alert")
async def receive_alert(request: Request):
    body = await request.json()
    print(f"Alert received: {json.dumps(body, indent=2)}")

    # Extract alert name from Grafana payload
    alerts = body.get("alerts", [])
    for alert in alerts:
        alert_name = alert.get("labels", {}).get("alertname", "")
        state      = alert.get("status", "")

        if state == "firing" and alert_name in ["LowFraudRecall", "DataDriftHigh"]:
            print(f"Triggering CI/CD for: {alert_name}")
            await trigger_github_actions(alert_name)

    return {"status": "received"}

async def trigger_github_actions(reason: str):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/cicd.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "ref": "main",
        "inputs": {"trigger_reason": reason}
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload)
        print(f"GitHub Actions response: {r.status_code}")
    return r.status_code

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
