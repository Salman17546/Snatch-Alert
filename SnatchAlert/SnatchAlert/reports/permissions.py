from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow the reporter of an incident to edit/delete it.
    """
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only to the reporter
        return obj.reported_by == request.user


class IsAdminOrAuthority(permissions.BasePermission):
    """
    Custom permission to only allow admin or authority users.
    """
    def has_permission(self, request, view):
        return request.user and request.user.is_authenticated and (
            request.user.role in ['admin', 'authority'] or request.user.is_staff
        )


class IsReporter(permissions.BasePermission):
    """
    Allow access only to the user who reported (victim.user).
    """
    def has_object_permission(self, request, view, obj):
        # obj is IncidentFact
        if not obj.victim:
            return False
        reporter = getattr(obj.victim, "user", None)
        return reporter is not None and reporter == request.user
