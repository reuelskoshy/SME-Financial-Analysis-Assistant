import chromadb, inspect
print('chromadb version:', getattr(chromadb, '__version__', 'unknown'))
print('\nMembers containing Client:')
print([m for m in dir(chromadb) if 'Client' in m or 'client' in m.lower()])
try:
    from chromadb.config import Settings
    print('\nSettings type:', type(Settings), 'signature:', inspect.signature(Settings))
except Exception as e:
    print('\nSettings import error:', e)
try:
    print('\nchromadb.Client signature:', inspect.signature(chromadb.Client))
except Exception as e:
    print('\nClient signature error:', e)
print('\nSample module attrs:', [a for a in dir(chromadb) if a.lower().startswith('persist') or 'duck' in a.lower()])
