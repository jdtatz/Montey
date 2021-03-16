use std::collections::BTreeMap;

use darling::FromMeta;
use dynfmt::{Format, SimpleCurlyFormat};
use itertools::Itertools;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, AttributeArgs, Ident, ItemFn, Lit, Type};

struct WrappedType(Type);

impl FromMeta for WrappedType {
    fn from_string(value: &str) -> darling::Result<Self> {
        syn::parse_str(value)
            .map(WrappedType)
            .map_err(|_| darling::Error::unknown_value(value))
    }

    fn from_value(value: &Lit) -> darling::Result<Self> {
        if let Lit::Str(ref lit_str) = *value {
            lit_str
                .parse()
                .map(WrappedType)
                .map_err(|_| darling::Error::unknown_value(&lit_str.value()).with_span(value))
        } else {
            Err(darling::Error::unexpected_lit_type(value))
        }
    }
}

fn unwrap_typed(wt: WrappedType) -> Type { wt.0 }

#[derive(Debug, FromMeta)]
struct SubstituteArg {
    #[darling(rename = "type", map = "unwrap_typed")]
    typed:  Type,
    format: String,
}

#[derive(Debug, FromMeta)]
struct GenericArg {
    param:      Ident,
    #[darling(multiple)]
    substitute: Vec<SubstituteArg>,
}

#[derive(Debug, FromMeta)]
struct MacroArgs {
    format:  String,
    #[darling(default)]
    abi:     Option<String>,
    #[darling(multiple)]
    generic: Vec<GenericArg>,
}

#[proc_macro_attribute]
pub fn generate_extern_wrapper(args: TokenStream, input: TokenStream) -> TokenStream {
    let attr_args = parse_macro_input!(args as AttributeArgs);
    let wrapped_func = parse_macro_input!(input as ItemFn);

    let ItemFn { attrs, sig, .. } = &wrapped_func;
    let generics_idents = sig
        .generics
        .type_params()
        .map(|typ| typ.ident.clone())
        .collect::<Vec<_>>();
    if let Some(_) = sig.generics.lifetimes().next() {
        panic!("Lifetimes are not supported")
    }
    if let Some(_) = sig.generics.const_params().next() {
        unimplemented!("Const generics are not yet implmented")
    }

    let generic_args = attr_args.as_slice();
    let MacroArgs {
        format,
        abi,
        generic: generics,
    } = match <MacroArgs as FromMeta>::from_list(generic_args) {
        Ok(ma) => ma,
        Err(e) => {
            return TokenStream::from(e.write_errors());
        },
    };
    let abi = abi.unwrap_or_else(|| "C".to_string());
    let subs = generics
        .into_iter()
        .map(|GenericArg { param, substitute }| {
            (
                param,
                substitute
                    .into_iter()
                    .map(|SubstituteArg { typed, format }| (typed, format))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        generics_idents.len(),
        subs.len(),
        "The number of substitutions is not equal to the numberr of generic arguments"
    );
    for g in generics_idents.iter() {
        assert!(subs.contains_key(g))
    }

    fn pat2ident(pat: &syn::Pat) -> Ident {
        match pat {
            syn::Pat::Ident(i) => i.ident.clone(),
            syn::Pat::Reference(r) => pat2ident(&r.pat),
            syn::Pat::Type(ty) => pat2ident(&ty.pat),
            p => unimplemented!("Function arg pattern {:?} is not implmented", p),
        }
    }

    let call_args = sig
        .inputs
        .iter()
        .map(|a| match a {
            syn::FnArg::Receiver(r) => unimplemented!("Self-like Receiver type {:?} is not implmentd", r),
            syn::FnArg::Typed(t) => pat2ident(&t.pat),
        })
        .collect::<Vec<_>>();

    let wrappers = subs
        .into_iter()
        .map(|(ty, h)| h.into_iter().map(|(k, v)| (ty.clone(), (k, v))).collect::<Vec<_>>())
        .multi_cartesian_product()
        .map(|mp| {
            let (tys, paths): (Vec<_>, Vec<_>) = mp.iter().map(|(ty, (p, _))| (ty, p)).unzip();
            let arguments = mp
                .iter()
                .map(|(ty, (_, sub))| (ty.to_string(), sub))
                .collect::<BTreeMap<_, _>>();
            let wrapped_name = SimpleCurlyFormat.format(&format, arguments).unwrap();
            let wrapped_name = format_ident!("{}", wrapped_name);
            let name = &sig.ident;
            let inputs = &sig.inputs;
            let output = &sig.output;
            quote! {
                mod #wrapped_name {
                    use super::*;

                    #( type #tys = #paths; )*

                    #[no_mangle]
                    #( #attrs )*
                    unsafe extern #abi fn #wrapped_name( #inputs ) #output {
                        #name :: < #( #generics_idents ),* > ( #( #call_args ),* )
                    }
                }
            }
        });

    let res = quote! {
        #wrapped_func
        #( #wrappers )*
    };
    // panic!("{}", res);

    TokenStream::from(res)
}
