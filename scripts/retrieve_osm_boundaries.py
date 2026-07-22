# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Retrieve administrative boundaries for the specified country using the overpass API and save it
to the specified output files.

Note that overpass requests are based on a fair
use policy. `retrieve_osm_data` is meant to be used in a way that respects this
policy by fetching the needed data once, only.
"""

import json
import logging
import time

import requests

from scripts._helpers import (  # set_scenario_config,; update_config_from_wildcards,; update_config_from_wildcards,
    configure_logging,
    set_scenario_config,
)

logger = logging.getLogger(__name__)


ADM1_SPECIALS = {
    "XK": 5,
}


def retrieve_osm_boundaries(
    country,
    adm1_specials,
    output,
    url="https://overpass-api.de/api/interpreter",
    max_tries=3,
    timeout=600,
    user_agent="",
):
    """
    Retrieve OSM administrative boundaries for the specified country and save it to the specified
    output files.

    Parameters
    ----------
    country : str
        The country code for which the OSM data should be retrieved.
    url : str, optional
        The URL of the overpass API endpoint. The default is
        "https://overpass-api.de/api/interpreter".
    max_tries : int, optional
        The maximum number of attempts to retrieve the data in case of failure.
        The default is 3.
    timeout : int, optional
        The timeout in seconds for the overpass API requests. The default is 600.
    user_agent : str
        The User-Agent string to include in the request headers for fair use
        policy compliance. Note that overpass-api.de answers requests carrying a
        default library agent (python-requests/*, curl/*) with HTTP 406.
    """
    wait_time = 5

    headers = {"User-Agent": user_agent}

    osm_adm_level = "4"
    if country in adm1_specials:
        osm_adm_level = adm1_specials[country]  # special case e.g. for Kosovo

    retries = max_tries
    for attempt in range(retries):
        logger.info(
            f" - Fetching OSM administrative boundaries for {country} (Attempt {attempt + 1})..."
        )

        # Build the overpass query
        op_area = f'area["ISO3166-1"="{country}"]'
        op_query = f"""
            [out:json][timeout:{timeout}];
            {op_area}->.searchArea;
            (
            relation["boundary"="administrative"]["admin_level"={osm_adm_level}]["name"](area.searchArea);
            );
            out body geom;
        """
        try:
            # Send the request
            response = requests.post(url, data=op_query, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses

            filepath = output[0]

            with open(filepath, mode="w") as f:
                json.dump(response.json(), f, indent=2)
            logger.info(" - Done.")
            break  # Exit the retry loop on success
        except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
            logger.error(
                f"Error for retrieving administrative boundaries in country {country}: {e}"
            )
            logger.debug(
                f"Response text: {response.text if response else 'No response'}"
            )
            if attempt < retries - 1:
                wait_time += 15
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to retrieve administrative boundaries in country {country} after {retries} attempts."
                )
        except Exception as e:
            # For now, catch any other exceptions and log them. Treat this
            # the same as a RequestException and try to run again two times.
            logger.error(
                f"Unexpected error in retrieving administrative boundaries in country {country}: {e}"
            )
            if attempt < retries - 1:
                wait_time += 10
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Failed to retrieve administrative boundaries for country {country} after {retries} attempts."
                )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_osm_boundaries", country="XK")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    overpass_api = snakemake.params.overpass_api

    # Build User-Agent header
    ua_cfg = overpass_api["user_agent"]
    user_agent = (
        f"{ua_cfg['project_name']} "
        f"(Contact: {ua_cfg['email']}; Website: {ua_cfg['website']})"
    )

    # Retrieve the OSM data
    country = snakemake.wildcards.country
    output = snakemake.output

    retrieve_osm_boundaries(
        country,
        ADM1_SPECIALS,
        output,
        url=overpass_api["url"],
        max_tries=overpass_api["max_tries"],
        timeout=overpass_api["timeout"],
        user_agent=user_agent,
    )
