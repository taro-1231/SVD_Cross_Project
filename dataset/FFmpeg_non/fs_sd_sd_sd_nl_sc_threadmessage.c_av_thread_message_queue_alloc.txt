int func1 ( AVThreadMessageQueue ** var1 , 
unsigned var2 , 
unsigned var3 ) 
{ 
AVThreadMessageQueue * var4 ; 
int var5 = number ; 
if ( var2 > var6 / var3 ) 
return func2 ( var7 ) ; 
if ( ! ( var4 = func3 ( sizeof ( * var4 ) ) ) ) 
return func2 ( var8 ) ; 
if ( ( var5 = func4 ( & var4 -> var9 , var10 ) ) ) { 
func5 ( var4 ) ; 
return func2 ( var5 ) ; 
} 
if ( ( var5 = func6 ( & var4 -> var11 , var10 ) ) ) { 
func7 ( & var4 -> var9 ) ; 
func5 ( var4 ) ; 
return func2 ( var5 ) ; 
} 
if ( ! ( var4 -> var12 = func8 ( var3 * var2 ) ) ) { 
func9 ( & var4 -> var11 ) ; 
func7 ( & var4 -> var9 ) ; 
func5 ( var4 ) ; 
return func2 ( var5 ) ; 
} 
var4 -> var3 = var3 ; 
* var1 = var4 ; 
return number ; 
* var1 = var10 ; 
return func2 ( ENOSYS ) ; 
} 
