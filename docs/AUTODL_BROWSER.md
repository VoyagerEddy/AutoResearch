# AutoDL browser adapter

AutoResearch uses the official AutoDL Pro API for normal provisioning. The optional browser adapter is a fallback for account setup and console workflows when a developer token or image UUID is not yet available. It launches an installed Chrome or Edge executable through Playwright; it does not download a separate browser.

Install the optional dependencies:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[browser]"
```

Run an interactive login:

```powershell
.venv/Scripts/python.exe -m autoresearch autodl-browser-login --browser auto --open console --keep-open
```

The same reusable login helper is available in **Settings & Connections → AutoDL Pro API**. Click **Configure AutoDL Login** once: the phone and password are stored in the current user's local application data and encrypted with Windows DPAPI. Neither value is written to this repository or `.env`. On later runs, click **Auto Login to AutoDL**; AutoResearch opens the page and fills the saved credentials automatically. Complete the visible security verification and enter the SMS one-time code yourself. Those one-time values are never saved. The dashboard launches this workflow in a separate interactive console, so there is no temporary browser-console script to paste on later logins.

The CLI prompts for the phone number, password, and SMS one-time code. For unattended credential injection, `AUTODL_BROWSER_PHONE` and `AUTODL_BROWSER_PASSWORD` may be supplied in the process environment. Do not add those variables to `.env`, shell history, source files, or logs. The adapter keeps the Playwright page and browser context alive while it waits for the one-time-code callback, then clears its in-memory references when the session closes.

The Aliyun puzzle step is automated locally. The solver downloads the background and transparent piece images inside the current browser context, matches the piece contour against image gradients, and converts the source offset to rendered pixels. It then follows a curved variable-speed mouse path and corrects the pointer while the button remains held by observing the puzzle piece's actual position. This closed-loop correction handles the component's velocity-dependent lag. A challenge refresh is treated as a failed attempt and solved again with the new images.

Callers can continue with an authenticated instance workflow in the same process:

```python
import asyncio
import getpass

from autoresearch.services.autodl_browser import AutoDLBrowserSession, AutoDLCredentials


async def read_otp() -> str:
    return await asyncio.to_thread(input, "SMS one-time code: ")


async def inspect_instances(session: AutoDLBrowserSession) -> None:
    await session.open_instance_list()
    # Continue with reviewed, site-specific instance selection on session.page.


async def main() -> None:
    credentials = AutoDLCredentials(
        phone=input("AutoDL phone: "),
        password=getpass.getpass("AutoDL password: "),
    )
    async with AutoDLBrowserSession() as session:
        await session.run_authenticated(credentials, read_otp, inspect_instances)


asyncio.run(main())
```

`run_authenticated` accepts either a synchronous or asynchronous hook. `open_instance_list`, `open_instance_market`, and `goto_autodl` are the supported navigation seams for a later instance-selection service. Navigation is restricted to trusted AutoDL HTTPS hosts. Billable instance creation still needs the existing action-time confirmation before a final create click or API request.

If browser discovery fails, pass `--browser chrome`, `--browser edge`, or an absolute `--executable-path`. The adapter checks the command path and standard Windows installation directories.
