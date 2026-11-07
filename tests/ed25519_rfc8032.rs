//! Ed25519 (RFC 8032): known-answer vectors and malleability regressions.
//!
//! The malleability tests pin RFC 8032 section 5.1.7: a verifier MUST reject S >= L. Without that
//! check, a signature with S' = S + L verifies, so one signed message has two valid encodings.
use apeiron::map::hashtrinity::crypto::{ed25519_pubkey, ed25519_sign, ed25519_verify};

fn unhex<const N: usize>(s: &str) -> [u8; N] {
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// Group order L = 2^252 + 27742317777372353535851937790883648493, little-endian.
fn group_order_le() -> [u8; 32] {
    let limbs: [u64; 4] = [
        0x5812631a5cf5d3ed,
        0x14def9dea2f79cd6,
        0,
        0x1000000000000000,
    ];
    let mut out = [0u8; 32];
    for (i, l) in limbs.iter().enumerate() {
        out[i * 8..i * 8 + 8].copy_from_slice(&l.to_le_bytes());
    }
    out
}

/// a + b over 256-bit little-endian integers; None on overflow.
fn add_le(a: &[u8; 32], b: &[u8; 32]) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    let mut carry = 0u16;
    for i in 0..32 {
        let v = a[i] as u16 + b[i] as u16 + carry;
        out[i] = v as u8;
        carry = v >> 8;
    }
    (carry == 0).then_some(out)
}

/// The same signature with its S half replaced by `s`.
fn with_s(sig: &[u8; 64], s: &[u8; 32]) -> [u8; 64] {
    let mut out = *sig;
    out[32..].copy_from_slice(s);
    out
}

fn s_half(sig: &[u8; 64]) -> [u8; 32] {
    sig[32..].try_into().unwrap()
}

const TEST1_SK: &str = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";
const TEST1_PK: &str = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
const TEST1_SIG: &str = "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b";
const TEST2_SK: &str = "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb";
const TEST2_PK: &str = "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c";
const TEST2_MSG: [u8; 1] = [0x72];
const TEST2_SIG: &str = "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00";

#[test]
fn rfc8032_test_vectors() {
    let sk1: [u8; 32] = unhex(TEST1_SK);
    assert_eq!(ed25519_pubkey(&sk1), unhex::<32>(TEST1_PK));
    let sig1 = ed25519_sign(&sk1, b"");
    assert_eq!(sig1, unhex::<64>(TEST1_SIG));
    assert!(ed25519_verify(&unhex(TEST1_PK), b"", &sig1));

    let sk2: [u8; 32] = unhex(TEST2_SK);
    assert_eq!(ed25519_pubkey(&sk2), unhex::<32>(TEST2_PK));
    let sig2 = ed25519_sign(&sk2, &TEST2_MSG);
    assert_eq!(sig2, unhex::<64>(TEST2_SIG));
    assert!(ed25519_verify(&unhex(TEST2_PK), &TEST2_MSG, &sig2));
}

#[test]
fn rejects_tampered_message_and_signature() {
    let pk: [u8; 32] = unhex(TEST1_PK);
    let sig: [u8; 64] = unhex(TEST1_SIG);
    assert!(!ed25519_verify(&pk, b"x", &sig));
    let mut bad = sig;
    bad[10] ^= 0x01; // flip a bit in R
    assert!(!ed25519_verify(&pk, b"", &bad));
}

#[test]
fn rejects_s_plus_l_malleation() {
    // RFC 8032 section 5.1.7 step 1: S must be < L. S + L encodes the same scalar mod L.
    let pk: [u8; 32] = unhex(TEST1_PK);
    let sig: [u8; 64] = unhex(TEST1_SIG);
    assert!(ed25519_verify(&pk, b"", &sig));
    let s2 = add_le(&s_half(&sig), &group_order_le()).expect("S + L fits in 256 bits");
    let malleated = with_s(&sig, &s2);
    assert_ne!(malleated, sig);
    assert!(
        !ed25519_verify(&pk, b"", &malleated),
        "S + L malleation must be rejected"
    );
}

#[test]
fn rejects_s_plus_2l_malleation() {
    let pk: [u8; 32] = unhex(TEST2_PK);
    let sig: [u8; 64] = unhex(TEST2_SIG);
    assert!(ed25519_verify(&pk, &TEST2_MSG, &sig));
    let l = group_order_le();
    let s1 = add_le(&s_half(&sig), &l).expect("fits");
    let s2 = add_le(&s1, &l).expect("fits");
    assert!(
        !ed25519_verify(&pk, &TEST2_MSG, &with_s(&sig, &s2)),
        "S + 2L malleation must be rejected"
    );
}

#[test]
fn rejects_s_equal_to_l() {
    // S = L is not a canonical scalar (it reduces to 0), so it must not verify even with a valid R.
    let pk: [u8; 32] = unhex(TEST1_PK);
    let sig: [u8; 64] = unhex(TEST1_SIG);
    assert!(!ed25519_verify(&pk, b"", &with_s(&sig, &group_order_le())));
}
