using System.Collections.Generic;
using System.ComponentModel;
using System.Threading.Tasks;
using Godot;
using Godot.Bridge;
using Godot.NativeInterop;
using MegaCrit.Sts2.Core.Assets;
using MegaCrit.Sts2.Core.Helpers;
using MegaCrit.Sts2.Core.HoverTips;
using MegaCrit.Sts2.Core.Localization;
using MegaCrit.Sts2.Core.Models.Badges;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.HoverTips;
using MegaCrit.Sts2.Core.TestSupport;

namespace MegaCrit.Sts2.Core.Nodes.Screens.GameOverScreen;

[ScriptPath("res://src/Core/Nodes/Screens/GameOverScreen/NBadge.cs")]
public class NBadge : NButton
{
	public new class MethodName : NButton.MethodName
	{
		public static readonly StringName Create = "Create";

		public new static readonly StringName _Ready = "_Ready";

		public static readonly StringName GetRarityPrefix = "GetRarityPrefix";

		public new static readonly StringName _ExitTree = "_ExitTree";

		public new static readonly StringName OnFocus = "OnFocus";

		public new static readonly StringName OnUnfocus = "OnUnfocus";

		public static readonly StringName GetBadgeBaseTexture = "GetBadgeBaseTexture";
	}

	public new class PropertyName : NButton.PropertyName
	{
		public static readonly StringName _tween = "_tween";

		public static readonly StringName _hoverNode = "_hoverNode";
	}

	public new class SignalName : NButton.SignalName
	{
	}

	private static readonly string _scenePath = SceneHelper.GetScenePath("screens/game_over_screen/badge");

	private Tween? _tween;

	private HoverTip _hoverTip;

	private LocString _title;

	private LocString _description;

	private const string _table = "badges";

	private Control _hoverNode;

	public static NBadge? Create(Badge badgeModel)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBadge nBadge = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBadge>(PackedScene.GenEditState.Disabled);
		nBadge.GetNode<TextureRect>("%BadgeHolder").Texture = badgeModel.BadgeBase;
		nBadge.GetNode<TextureRect>("%Icon").Texture = badgeModel.BadgeIcon;
		if (LocString.Exists("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Title"))
		{
			nBadge._title = new LocString("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Title");
			nBadge._description = new LocString("badges", badgeModel.Id + "." + GetRarityPrefix(badgeModel.Rarity) + "Description");
		}
		else
		{
			nBadge._title = new LocString("badges", badgeModel.Id + ".title");
			nBadge._description = new LocString("badges", badgeModel.Id + ".description");
		}
		return nBadge;
	}

	public static NBadge? Create(string id, BadgeRarity rarity)
	{
		if (TestMode.IsOn)
		{
			return null;
		}
		NBadge nBadge = PreloadManager.Cache.GetScene(_scenePath).Instantiate<NBadge>(PackedScene.GenEditState.Disabled);
		nBadge.GetNode<TextureRect>("%BadgeHolder").Texture = GetBadgeBaseTexture(rarity);
		nBadge.GetNode<TextureRect>("%Icon").Texture = PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_" + id.ToLowerInvariant() + ".png"));
		if (LocString.Exists("badges", id + "." + GetRarityPrefix(rarity) + "Title"))
		{
			nBadge._title = new LocString("badges", id + "." + GetRarityPrefix(rarity) + "Title");
			nBadge._description = new LocString("badges", id + "." + GetRarityPrefix(rarity) + "Description");
		}
		else
		{
			nBadge._title = new LocString("badges", id + ".title");
			nBadge._description = new LocString("badges", id + ".description");
		}
		return nBadge;
	}

	public override void _Ready()
	{
		ConnectSignals();
		_hoverNode = this;
		_hoverTip = new HoverTip(_title, _description);
	}

	private static string GetRarityPrefix(BadgeRarity rarity)
	{
		return rarity switch
		{
			BadgeRarity.Bronze => "bronze", 
			BadgeRarity.Silver => "silver", 
			BadgeRarity.Gold => "gold", 
			_ => "ERROR", 
		};
	}

	public async Task AnimateIn()
	{
		_tween = CreateTween().SetParallel();
		_tween.TweenProperty(this, "modulate:a", 1f, 0.25);
		_tween.TweenProperty(this, "position:y", base.Position.Y, 0.25).SetEase(Tween.EaseType.Out).SetTrans(Tween.TransitionType.Spring)
			.From(base.Position.Y + 40f);
		await ToSignal(_tween, Tween.SignalName.Finished);
	}

	public override void _ExitTree()
	{
		_tween?.Kill();
	}

	protected override void OnFocus()
	{
		base.OnFocus();
		NHoverTipSet nHoverTipSet = NHoverTipSet.CreateAndShow(_hoverNode, _hoverTip);
		nHoverTipSet.GlobalPosition = _hoverNode.GlobalPosition + new Vector2(10f, -132f);
	}

	protected override void OnUnfocus()
	{
		base.OnUnfocus();
		NHoverTipSet.Remove(_hoverNode);
	}

	private static Texture2D GetBadgeBaseTexture(BadgeRarity rarity)
	{
		return rarity switch
		{
			BadgeRarity.Bronze => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_bronze.png")), 
			BadgeRarity.Silver => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_silver.png")), 
			BadgeRarity.Gold => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("ui/game_over_screen/badge_gold.png")), 
			_ => PreloadManager.Cache.GetTexture2D(ImageHelper.GetImagePath("atlases/power_atlas.sprites/missing_power.tres")), 
		};
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	internal new static List<MethodInfo> GetGodotMethodList()
	{
		List<MethodInfo> list = new List<MethodInfo>(7);
		list.Add(new MethodInfo(MethodName.Create, new PropertyInfo(Variant.Type.Object, "", PropertyHint.None, "", PropertyUsageFlags.Default, new StringName("Control"), exported: false), MethodFlags.Normal | MethodFlags.Static, new List<PropertyInfo>
		{
			new PropertyInfo(Variant.Type.String, "id", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false),
			new PropertyInfo(Variant.Type.Int, "rarity", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false)
		}, null));
		list.Add(new MethodInfo(MethodName._Ready, new PropertyInfo(Variant.Type.Nil, "", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false), MethodFlags.Normal, null, null));
		list.Add(new MethodInfo(MethodName.GetRarityPrefix, new PropertyInfo(Variant.Type.String, "", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false), MethodFlags.Normal | MethodFlags.Static, new List<PropertyInfo>
		{
			new PropertyInfo(Variant.Type.Int, "rarity", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false)
		}, null));
		list.Add(new MethodInfo(MethodName._ExitTree, new PropertyInfo(Variant.Type.Nil, "", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false), MethodFlags.Normal, null, null));
		list.Add(new MethodInfo(MethodName.OnFocus, new PropertyInfo(Variant.Type.Nil, "", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false), MethodFlags.Normal, null, null));
		list.Add(new MethodInfo(MethodName.OnUnfocus, new PropertyInfo(Variant.Type.Nil, "", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false), MethodFlags.Normal, null, null));
		list.Add(new MethodInfo(MethodName.GetBadgeBaseTexture, new PropertyInfo(Variant.Type.Object, "", PropertyHint.None, "", PropertyUsageFlags.Default, new StringName("Texture2D"), exported: false), MethodFlags.Normal | MethodFlags.Static, new List<PropertyInfo>
		{
			new PropertyInfo(Variant.Type.Int, "rarity", PropertyHint.None, "", PropertyUsageFlags.Default, exported: false)
		}, null));
		return list;
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override bool InvokeGodotClassMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
	{
		if (method == MethodName.Create && args.Count == 2)
		{
			ret = VariantUtils.CreateFrom<NBadge>(Create(VariantUtils.ConvertTo<string>(in args[0]), VariantUtils.ConvertTo<BadgeRarity>(in args[1])));
			return true;
		}
		if (method == MethodName._Ready && args.Count == 0)
		{
			_Ready();
			ret = default(godot_variant);
			return true;
		}
		if (method == MethodName.GetRarityPrefix && args.Count == 1)
		{
			ret = VariantUtils.CreateFrom<string>(GetRarityPrefix(VariantUtils.ConvertTo<BadgeRarity>(in args[0])));
			return true;
		}
		if (method == MethodName._ExitTree && args.Count == 0)
		{
			_ExitTree();
			ret = default(godot_variant);
			return true;
		}
		if (method == MethodName.OnFocus && args.Count == 0)
		{
			OnFocus();
			ret = default(godot_variant);
			return true;
		}
		if (method == MethodName.OnUnfocus && args.Count == 0)
		{
			OnUnfocus();
			ret = default(godot_variant);
			return true;
		}
		if (method == MethodName.GetBadgeBaseTexture && args.Count == 1)
		{
			ret = VariantUtils.CreateFrom<Texture2D>(GetBadgeBaseTexture(VariantUtils.ConvertTo<BadgeRarity>(in args[0])));
			return true;
		}
		return base.InvokeGodotClassMethod(in method, args, out ret);
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	internal static bool InvokeGodotClassStaticMethod(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret)
	{
		if (method == MethodName.Create && args.Count == 2)
		{
			ret = VariantUtils.CreateFrom<NBadge>(Create(VariantUtils.ConvertTo<string>(in args[0]), VariantUtils.ConvertTo<BadgeRarity>(in args[1])));
			return true;
		}
		if (method == MethodName.GetRarityPrefix && args.Count == 1)
		{
			ret = VariantUtils.CreateFrom<string>(GetRarityPrefix(VariantUtils.ConvertTo<BadgeRarity>(in args[0])));
			return true;
		}
		if (method == MethodName.GetBadgeBaseTexture && args.Count == 1)
		{
			ret = VariantUtils.CreateFrom<Texture2D>(GetBadgeBaseTexture(VariantUtils.ConvertTo<BadgeRarity>(in args[0])));
			return true;
		}
		ret = default(godot_variant);
		return false;
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override bool HasGodotClassMethod(in godot_string_name method)
	{
		if (method == MethodName.Create)
		{
			return true;
		}
		if (method == MethodName._Ready)
		{
			return true;
		}
		if (method == MethodName.GetRarityPrefix)
		{
			return true;
		}
		if (method == MethodName._ExitTree)
		{
			return true;
		}
		if (method == MethodName.OnFocus)
		{
			return true;
		}
		if (method == MethodName.OnUnfocus)
		{
			return true;
		}
		if (method == MethodName.GetBadgeBaseTexture)
		{
			return true;
		}
		return base.HasGodotClassMethod(in method);
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
	{
		if (name == PropertyName._tween)
		{
			_tween = VariantUtils.ConvertTo<Tween>(in value);
			return true;
		}
		if (name == PropertyName._hoverNode)
		{
			_hoverNode = VariantUtils.ConvertTo<Control>(in value);
			return true;
		}
		return base.SetGodotClassPropertyValue(in name, in value);
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
	{
		if (name == PropertyName._tween)
		{
			value = VariantUtils.CreateFrom(in _tween);
			return true;
		}
		if (name == PropertyName._hoverNode)
		{
			value = VariantUtils.CreateFrom(in _hoverNode);
			return true;
		}
		return base.GetGodotClassPropertyValue(in name, out value);
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	internal new static List<PropertyInfo> GetGodotPropertyList()
	{
		List<PropertyInfo> list = new List<PropertyInfo>();
		list.Add(new PropertyInfo(Variant.Type.Object, PropertyName._tween, PropertyHint.None, "", PropertyUsageFlags.ScriptVariable, exported: false));
		list.Add(new PropertyInfo(Variant.Type.Object, PropertyName._hoverNode, PropertyHint.None, "", PropertyUsageFlags.ScriptVariable, exported: false));
		return list;
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override void SaveGodotObjectData(GodotSerializationInfo info)
	{
		base.SaveGodotObjectData(info);
		info.AddProperty(PropertyName._tween, Variant.From(in _tween));
		info.AddProperty(PropertyName._hoverNode, Variant.From(in _hoverNode));
	}

	[EditorBrowsable(EditorBrowsableState.Never)]
	protected override void RestoreGodotObjectData(GodotSerializationInfo info)
	{
		base.RestoreGodotObjectData(info);
		if (info.TryGetProperty(PropertyName._tween, out var value))
		{
			_tween = value.As<Tween>();
		}
		if (info.TryGetProperty(PropertyName._hoverNode, out var value2))
		{
			_hoverNode = value2.As<Control>();
		}
	}
}
