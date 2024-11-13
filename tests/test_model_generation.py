from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from functioncalming import get_completion


@pytest.mark.asyncio
async def test_defaults_work():
    class File(BaseModel):
        file_name: str
        file_content: str = "blank"

    class Folder(BaseModel):
        folder_name: str
        contents: "list[Folder | File]" = []

    class Result(BaseModel):
        result: Folder | File

    File.model_rebuild()
    Folder.model_rebuild()

    calm_response = await get_completion(
        system_prompt=None,
        user_message="Create an file named hello.txt that is inside of two nested folders. Leave the file empty (default content).",
        tools=[Result],
        retries=1,
        model="gpt-4o-mini"
    )

    assert calm_response.tool_call_results[0].result.contents[0].contents[0].file_name == "hello.txt"
    assert calm_response.tool_call_results[0].result.contents[0].contents[0].file_content == "blank"

