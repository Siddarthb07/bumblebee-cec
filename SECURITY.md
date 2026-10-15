# Security policy

Bumblebee is a local closed-corpus correlator. It is not a network service,
and it does not ship credentials.

## Reporting a vulnerability

Please use [private vulnerability reporting](https://github.com/Siddarthb07/bumblebee-cec/security/advisories/new)
on this repository, or email [siddarthb078@gmail.com](mailto:siddarthb078@gmail.com).

Include what you ran, what you expected, and what happened. Do not open a
public issue for an unfixed vulnerability.

## What is in scope

- The CLI, the local dash (`127.0.0.1` only), and the training and eval code
  in this repository.
- Unsafe-topic refusal in `src/bumblebee/safety.py` failing open on a banned request.

## What is out of scope

- Upstream dataset content.
- Live tool results from public HTTP APIs (prices, weather). Those responses
  are unlabeled third-party data.
- A missing model checkpoint. That is a local setup state, not a vulnerability.
