%{
#include <stdio.h>
#include <stdlib.h>

extern char *yytext;
extern int yylineno;

int yylex();
void yyerror(const char *s) { fprintf(stderr, "[Error] (line %d) %s\n", yylineno, s); exit(1); }
%}

%locations
%error-verbose

%union {
	char *string_val;
}

%token '(' ')' '{' '}' '<' '>'
%token tI64 tSTRING tLIST tTABLE
%token tMODULE tIMPORT tDEF tCHECKCAST tRETURN
%token <string_val> tSTRINGVAL
%token <string_val> tFUNCTION
%token <string_val> tSYMBOL
%token <string_val> tIDENTIFIER

%start program

%%

program : tMODULE tIDENTIFIER '{' modulecontents '}'
	;

modulecontents : modulecontents modulecontent
	       | %empty
               ;

modulecontent : tDEF tIDENTIFIER '(' ')' ':' type '{' statements '}'
	      | tIMPORT tIDENTIFIER ';'
              ;

type : '?'
     | tI64
     | tSTRING
     | tTABLE
     | tLIST '<' type '>'
     ;

statements : statements statement
	   | %empty
           ;

statement : tIDENTIFIER ':' type '=' tCHECKCAST '(' call ',' type ')' ';'
          | tIDENTIFIER ':' type '=' call ';'
          | tIDENTIFIER ':' type '=' literal ';'
          | tRETURN tIDENTIFIER ';'
          ;

call : tFUNCTION '(' literals ')'
     ;

literals : literalsne
         | %empty
         ;

literalsne : literalsne ','  literal
           | literal
           ;

literal : tIDENTIFIER
	| tSYMBOL
        | tSTRINGVAL ':' tSTRING
        | '(' stringliterals ')' ':' tSTRING
        ;

stringliterals : stringliterals ',' tSTRINGVAL
	       | tSTRINGVAL
               ;


%%
