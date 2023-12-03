import io

import pytest
from pydantic import BaseModel

from functioncalming import distillery, get_completion
from functioncalming.utils import json_dump_messages


class Extraction(BaseModel):
    data: dict


original_description = "Delegated extraction pipeline"
distil_function_rename = "register_extracted_data"
distil_function_descript = "Callback function"


@distillery(function_name=distil_function_rename, function_description=distil_function_descript)
async def extraction_pipeline(user_message: str) -> Extraction:
    f"""
    {original_description}
    """
    # call an expensive extraction pipeline on the data...
    return Extraction(data={"result": "hello world"})


@pytest.mark.asyncio
async def test_distillery_call():
    system_prompt = "To extract data for the user, delegate the task via a call to the extraction pipeline"
    distil_prompt = "Extract the data for the user and register the result using the provided function"
    dirty_history = []
    with io.StringIO() as fake_file:
        result_objects, clean_history = await get_completion(
            messages=dirty_history,  # usually, this is not needed - here we do this just to compare the two histories
            system_prompt=system_prompt,
            user_message="Hey there",
            rewrite_system_prompt_to=distil_prompt,
            tools=[extraction_pipeline],
            rewrite_history=False,
            rewrite_log_destination=fake_file
        )
        file_content = fake_file.getvalue()
    assert len(dirty_history)
    assert len(clean_history)
    assert clean_history != dirty_history
    clean_hist_str, dirty_hist_str = json_dump_messages(clean_history), json_dump_messages(dirty_history)
    # check that prompts are replaced
    assert system_prompt in dirty_hist_str
    assert distil_prompt not in dirty_hist_str
    assert distil_prompt in clean_hist_str
    assert system_prompt not in clean_hist_str

    # check that function name is replaced
    assert distil_function_rename in file_content
    assert extraction_pipeline.__name__ not in file_content

    # check that function description is replaced
    assert distil_function_descript in file_content
    assert original_description not in file_content

