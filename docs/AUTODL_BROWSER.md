# AutoDL browser adapter

AutoResearch uses the official AutoDL Pro API for provisioning requested through ChatGPT. The optional local CLI adapter supports AutoDL account setup and troubleshooting. It uses an installed Chrome or Edge browser through Playwright.

Install the optional dependencies:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[browser]"
```

Run the local SMS login helper for AutoDL account troubleshooting.

```powershell
.venv/Scripts/python.exe -m autoresearch autodl-browser-login --sms --browser auto --open console --keep-open
```

The SMS helper uses `AUTODL_BROWSER_PHONE` from its process environment when available. It requests one SMS code and reads it in the private terminal. The password flow also accepts `AUTODL_BROWSER_PASSWORD` from the process environment. Keep those variables out of `.env`, shell history, source files, and logs. The adapter keeps the browser open while it waits for the code and clears its in-memory references when the session closes. Pass `--manual-captcha` to troubleshoot slider verification manually.

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
